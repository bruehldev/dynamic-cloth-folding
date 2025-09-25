#!/usr/bin/env python3
import argparse, os, sys, json, inspect, hashlib
from pathlib import Path
import numpy as np

from env.cloth_env import ClothEnv
from rlkit.envs import wrappers
from utils import general_utils

def load_variant_safely():
    saved = sys.argv[:]
    try:
        sys.argv = [saved[0]]
        args = general_utils.argsparser()
        return general_utils.get_variant(args)
    finally:
        sys.argv = saved

def make_env(variant, engine=None):
    env_kwargs = dict(variant['env_kwargs'])
    if engine is not None:
        sig = inspect.signature(ClothEnv.__init__)
        params = set(sig.parameters.keys())
        for k in ('engine', 'physics_engine', 'backend'):
            if k in params:
                env_kwargs[k] = engine
                break
    base = ClothEnv(**env_kwargs, randomization_kwargs=variant['randomization_kwargs'])
    wrapped = wrappers.NormalizedBoxEnv(base)
    return general_utils.get_randomized_env(wrapped, randomization_kwargs=variant['randomization_kwargs'])

def sha1(arr: np.ndarray) -> str:
    m = hashlib.sha1()
    m.update(arr.tobytes())
    return m.hexdigest()

def describe_obs_sequence(env, steps: int = 10, rng=None):
    """Reset + N Steps mit Zufallsaktionen; sammelt pro Key dtype/shape/min/max/mean.
       Für 'image' zusätzlich SHA1 der ersten Beobachtung + 5 Stichproben-Pixel."""
    obs = env.reset()
    if rng is None:
        rng = np.random.RandomState(123)
    action_dim = int(np.prod(env.action_space.shape))

    # Aggregatoren
    info = {}
    first_image_digest = None
    sampled_pixels = None

    # Init aggregierte Stat-Struktur
    keys = list(obs.keys()) if isinstance(obs, dict) else ['__flat__']
    agg = {k: {"count": 0, "min": +np.inf, "max": -np.inf, "sum": 0.0, "sum2": 0.0,
               "shape": None, "dtype": None} for k in keys}

    def update_stats(k, arr):
        a = np.asarray(arr)
        st = agg[k]
        st["count"] += a.size
        st["min"] = float(min(st["min"], np.min(a)))
        st["max"] = float(max(st["max"], np.max(a)))
        st["sum"] += float(np.sum(a))
        st["sum2"] += float(np.sum(a * a))
        st["shape"] = tuple(a.shape)
        st["dtype"] = str(a.dtype)

    # erste Beobachtung auswerten
    if isinstance(obs, dict):
        for k, v in obs.items():
            update_stats(k, v)
            if k == "image" and first_image_digest is None:
                first_image_digest = sha1(np.asarray(v))
                flat = np.asarray(v).ravel()
                rng_idx = rng.choice(flat.size, size=min(5, flat.size), replace=False)
                sampled_pixels = {int(i): float(flat[i]) for i in rng_idx}
    else:
        update_stats('__flat__', obs)

    for _ in range(steps):
        a = rng.uniform(low=-1.0, high=1.0, size=(action_dim,)).astype(np.float32)
        a = a.reshape(env.action_space.shape)
        obs, r, done, info = env.step(a)
        if isinstance(obs, dict):
            for k, v in obs.items():
                update_stats(k, v)
        else:
            update_stats('__flat__', obs)
        if done:
            obs = env.reset()

    # Mittelwerte/Std
    for k, st in agg.items():
        n = max(st["count"], 1)
        mean = st["sum"] / n
        var = max(st["sum2"] / n - mean * mean, 0.0)
        st["mean"] = mean
        st["std"] = float(np.sqrt(var))
        # Aufräumen
        del st["sum"], st["sum2"], st["count"]

    return {
        "keys": keys,
        "stats": agg,
        "image": {
            "first_sha1": first_image_digest,
            "sampled_pixels": sampled_pixels
        } if first_image_digest is not None else None
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-a", required=True)
    ap.add_argument("--engine-b", required=True)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--save-dir", default="trainings/obs_parity")
    args, _ = ap.parse_known_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    variant = load_variant_safely()

    env_a = make_env(variant, engine=args.engine_a)
    env_b = make_env(variant, engine=args.engine_b)

    rng = np.random.RandomState(2024)
    rep_a = describe_obs_sequence(env_a, steps=args.steps, rng=np.random.RandomState(2024))
    rep_b = describe_obs_sequence(env_b, steps=args.steps, rng=np.random.RandomState(2024))

    # einfache Vergleichszusammenfassung
    summary = {"keys_equal": rep_a["keys"] == rep_b["keys"], "per_key": {}}
    keys = sorted(set(rep_a["keys"]) | set(rep_b["keys"]))
    for k in keys:
        sa = rep_a["stats"].get(k)
        sb = rep_b["stats"].get(k)
        if sa is None or sb is None:
            summary["per_key"][k] = {"present_in_a": sa is not None, "present_in_b": sb is not None}
            continue
        summary["per_key"][k] = {
            "shape_equal": sa["shape"] == sb["shape"],
            "dtype_equal": sa["dtype"] == sb["dtype"],
            "mean_delta": (sa["mean"] - sb["mean"]) if (sa and sb) else None,
            "std_delta":  (sa["std"] - sb["std"]) if (sa and sb) else None,
            "min_delta":  (sa["min"] - sb["min"]) if (sa and sb) else None,
            "max_delta":  (sa["max"] - sb["max"]) if (sa and sb) else None,
        }

    # Images: SHA1-Vergleich
    img_cmp = None
    if rep_a.get("image") and rep_b.get("image"):
        img_cmp = {
            "sha1_equal": rep_a["image"]["first_sha1"] == rep_b["image"]["first_sha1"],
            "sha1_a": rep_a["image"]["first_sha1"],
            "sha1_b": rep_b["image"]["first_sha1"]
        }

    out = {
        "engine_a": args.engine_a,
        "engine_b": args.engine_b,
        "steps": args.steps,
        "report_a": rep_a,
        "report_b": rep_b,
        "summary": summary,
        "image_compare": img_cmp
    }
    out_path = os.path.join(args.save_dir, f"obs_parity_{args.engine_a}_vs_{args.engine_b}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
