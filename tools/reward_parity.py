#!/usr/bin/env python3
import argparse, os, sys, json, hashlib, inspect
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

def rollout(env, episodes, max_path_length, rng):
    """Sammelt (s, a, r, done, success) Sequenzen mit fixen zufälligen Aktionen."""
    all_rewards, all_done, all_success = [], [], []
    # deterministische Action-Folge basierend auf RNG; wir speichern die seeds pro Schritt
    action_dim = int(np.prod(env.action_space.shape))
    for ep in range(episodes):
        obs = env.reset()
        ep_rewards, ep_done, ep_success = [], [], []
        for t in range(max_path_length):
            # deterministische Zufallsaktion
            a = rng.uniform(low=-1.0, high=1.0, size=(action_dim,)).astype(np.float32)
            a = a.reshape(env.action_space.shape)
            obs, r, done, info = env.step(a)
            ep_rewards.append(float(r))
            ep_done.append(bool(done))
            # robust: 'is_success' kann fehlen -> None
            s = info.get('is_success', None)
            if isinstance(s, (np.bool_, bool)):
                s = bool(s)
            ep_success.append(s)
            if done:
                # auch Truncation markieren, falls Gym TimeLimit gesetzt hat
                break
        all_rewards.append(ep_rewards)
        all_done.append(ep_done)
        all_success.append(ep_success)
    return all_rewards, all_done, all_success

def stats_delta(a, b):
    d = np.array(b, dtype=np.float64) - np.array(a, dtype=np.float64)
    return {
        "delta_mean": float(np.mean(d)),
        "delta_std": float(np.std(d)),
        "delta_max_abs": float(np.max(np.abs(d))) if d.size else 0.0
    }

def confusion_counts(x, y):
    """Konfusionsmatrix für booleans (None wird ignoriert)."""
    tp = tn = fp = fn = 0
    n = 0
    for xi, yi in zip(x, y):
        if xi is None or yi is None:
            continue
        n += 1
        if xi and yi: tp += 1
        elif (not xi) and (not yi): tn += 1
        elif (not xi) and yi: fp += 1
        elif xi and (not yi): fn += 1
    return {"n_compared": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn}

def flatten(lstlst):
    return [x for lst in lstlst for x in lst]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-path-length", type=int, default=50)
    ap.add_argument("--engine-a", type=str, required=True)
    ap.add_argument("--engine-b", type=str, required=True)
    ap.add_argument("--save-dir", type=str, default="trainings/reward_parity")
    args, _ = ap.parse_known_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    variant = load_variant_safely()

    # Fixierte RNG für identische Aktionsfolgen
    rng = np.random.RandomState(12345)

    # Engine A
    env_a = make_env(variant, engine=args.engine_a)
    r_a, d_a, s_a = rollout(env_a, args.episodes, args.max_path_length, rng)

    # RNG muss identisch starten -> neue Instanz mit selben Seed
    rng = np.random.RandomState(12345)
    env_b = make_env(variant, engine=args.engine_b)
    r_b, d_b, s_b = rollout(env_b, args.episodes, args.max_path_length, rng)

    # Reward-Parität (auf gemeinsame Längen pro Episode limitiert)
    r_a_f, r_b_f = [], []
    d_a_f, d_b_f = [], []
    s_a_f, s_b_f = [], []

    for ra, rb, da, db, sa, sb in zip(r_a, r_b, d_a, d_b, s_a, s_b):
        n = min(len(ra), len(rb))
        r_a_f.extend(ra[:n]); r_b_f.extend(rb[:n])
        d_a_f.extend(da[:n]); d_b_f.extend(db[:n])
        s_a_f.extend(sa[:n]); s_b_f.extend(sb[:n])

    reward_delta = stats_delta(r_a_f, r_b_f)
    # Done-Parität (bool delta als 0/1-Metrik zusätzlich zählen)
    done_matches = int(np.sum(np.array(d_a_f, dtype=np.bool_) == np.array(d_b_f, dtype=np.bool_)))
    done_total = int(min(len(d_a_f), len(d_b_f)))

    # Success-Konfusion
    success_cm = confusion_counts(s_a_f, s_b_f)

    out = {
        "engine_a": args.engine_a,
        "engine_b": args.engine_b,
        "episodes": args.episodes,
        "max_path_length": args.max_path_length,
        "N_reward": len(r_a_f),
        "reward_parity": reward_delta,
        "done_parity": {
            "n_compared": done_total,
            "n_equal": done_matches,
            "fraction_equal": float(done_matches / done_total) if done_total else 1.0
        },
        "success_confusion": success_cm
    }

    out_path = os.path.join(args.save_dir, f"parity_{args.engine_a}_vs_{args.engine_b}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
