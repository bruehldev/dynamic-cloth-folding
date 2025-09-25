#!/usr/bin/env python3
import os, argparse, json
from pathlib import Path
import sys
import inspect
import numpy as np

from env.cloth_env import ClothEnv
from rlkit.envs import wrappers
from utils import general_utils


# -------------------------------
# Variant sicher laden (ohne Flag-Kollision)
# -------------------------------
def load_variant_safely():
    saved_argv = sys.argv[:]
    try:
        sys.argv = [saved_argv[0]]  # keine fremden CLI-Flags durchreichen
        args = general_utils.argsparser()
        variant = general_utils.get_variant(args)
    finally:
        sys.argv = saved_argv
    return variant


# -------------------------------
# Env-Bau (mit engine-Param-Erkennung)
# -------------------------------
def build_env(variant, engine=None, seed=0):
    env_kwargs = dict(variant['env_kwargs'])

    if engine is not None:
        # Nur setzen, wenn __init__ so einen Param akzeptiert
        sig = inspect.signature(ClothEnv.__init__)
        allowed = set(sig.parameters.keys())
        for k in ('engine', 'physics_engine', 'backend'):
            if k in allowed:
                env_kwargs[k] = engine
                break

    base = ClothEnv(**env_kwargs, randomization_kwargs=variant['randomization_kwargs'])
    wrapped = wrappers.NormalizedBoxEnv(base)
    env = general_utils.get_randomized_env(wrapped, randomization_kwargs=variant['randomization_kwargs'])

    # (Best effort) Seed setzen
    try:
        env.seed(seed)
    except Exception:
        pass
    return env


# -------------------------------
# Policy (deterministisch, Aktion = 0)
# -------------------------------
class ZeroPolicy:
    def __init__(self, action_space):
        self._act = np.zeros(action_space.shape, dtype=np.float32)
    def __call__(self, obs):
        return self._act


# -------------------------------
# Rollouts
# -------------------------------
def run_rollouts(env, policy, episodes, max_len):
    ep_stats = []        # pro Episode: {'return': float, 'success': bool, ...}
    trajectories = []    # pro Episode: {'rewards': [], 'infos': [dict,...]}

    for _ in range(episodes):
        obs = env.reset()
        total_r = 0.0
        path = {'rewards': [], 'infos': [], 'success': False}

        for t in range(max_len):
            a = policy(obs)
            step_out = env.step(a)

            # Gymnasium: (obs, reward, terminated, truncated, info)
            # Gym classic: (obs, reward, done, info)
            if len(step_out) == 5:
                nxt, r, terminated, truncated, info = step_out
                done = bool(terminated or truncated)
            else:
                nxt, r, done, info = step_out

            total_r += float(r)
            path['rewards'].append(float(r))
            path['infos'].append(info or {})
            obs = nxt
            if done:
                break

        # Erfolg aus Infos extrahieren (falls vorhanden)
        succ_vals = [i.get('is_success') for i in path['infos'] if isinstance(i, dict) and 'is_success' in i]
        success = bool(np.mean(succ_vals) > 0.5) if succ_vals else False
        path['success'] = success

        ep_stats.append({
            'return': total_r,
            'success': success,
        })
        trajectories.append(path)

    return ep_stats, trajectories


def summarize(values):
    arr = np.array(values, dtype=np.float64)
    return {
        'mean': float(arr.mean()) if arr.size else None,
        'std': float(arr.std()) if arr.size else None,
        'min': float(arr.min()) if arr.size else None,
        'max': float(arr.max()) if arr.size else None,
    }


# -------------------------------
# CLI
# -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=5)
    ap.add_argument('--max-path-length', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--engine-a', default='mujoco')
    ap.add_argument('--engine-b', default='pybullet')
    ap.add_argument('--save-dir', default='trainings/rollout_compare')
    ap.add_argument('--policy-checkpoint', default=None,
                    help='Optional: Pfad zu einer Policy (Platzhalter: aktuell ungenutzt, ZeroPolicy wird verwendet)')
    args, _ = ap.parse_known_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    variant = load_variant_safely()

    # ----- Env A -----
    env_a = build_env(variant, engine=args.engine_a, seed=args.seed)
    policy_a = ZeroPolicy(env_a.action_space)  # gleiche deterministische Policy
    stats_a, traj_a = run_rollouts(env_a, policy_a, args.episodes, args.max_path_length)

    # ----- Env B -----
    env_b = build_env(variant, engine=args.engine_b, seed=args.seed)
    policy_b = ZeroPolicy(env_b.action_space)  # identisch deterministisch
    stats_b, traj_b = run_rollouts(env_b, policy_b, args.episodes, args.max_path_length)

    # ----- Zusammenfassung & Deltas -----
    returns_a = [e['return'] for e in stats_a]
    returns_b = [e['return'] for e in stats_b]
    succ_a = [1.0 if e['success'] else 0.0 for e in stats_a]
    succ_b = [1.0 if e['success'] else 0.0 for e in stats_b]

    sum_a = {'return': summarize(returns_a), 'success': summarize(succ_a)}
    sum_b = {'return': summarize(returns_b), 'success': summarize(succ_b)}

    deltas = np.array(returns_a, dtype=np.float64) - np.array(returns_b, dtype=np.float64)
    delta_summary = {
        'mean': float(deltas.mean()) if deltas.size else None,
        'std': float(deltas.std()) if deltas.size else None,
        'max_abs': float(np.abs(deltas).max()) if deltas.size else None,
    }

    print("=== Engine A:", args.engine_a, "===")
    print(json.dumps(sum_a, indent=2))
    print("=== Engine B:", args.engine_b, "===")
    print(json.dumps(sum_b, indent=2))
    print(f"Δ Return mean={delta_summary['mean']:.6f}, std={delta_summary['std']:.6f}, max|Δ|={delta_summary['max_abs']:.6f}")

    # ----- Persistenz -----
    np.savez_compressed(
        os.path.join(args.save_dir, f'rollouts_{args.engine_a}.npz'),
        stats=np.array(stats_a, dtype=object), traj=np.array(traj_a, dtype=object)
    )
    np.savez_compressed(
        os.path.join(args.save_dir, f'rollouts_{args.engine_b}.npz'),
        stats=np.array(stats_b, dtype=object), traj=np.array(traj_b, dtype=object)
    )
    with open(os.path.join(args.save_dir, 'summary.json'), 'w') as f:
        json.dump({
            'engine_a': args.engine_a, 'engine_b': args.engine_b,
            'episodes': args.episodes, 'max_path_length': args.max_path_length, 'seed': args.seed,
            'sum_a': sum_a, 'sum_b': sum_b,
            'delta_return': delta_summary,
        }, f, indent=2)


if __name__ == '__main__':
    main()
