# tools/script_policy_success.py
#!/usr/bin/env python3
import numpy as np, argparse, sys
from utils import general_utils
from env.cloth_env import ClothEnv
from rlkit.envs import wrappers
import os

def apply_easy_mode(variant):
    if os.getenv("EASY_MODE", "0") != "1":
        return variant
    v = dict(variant)
    # 1) Randomization aus
    rk = dict(v['randomization_kwargs'])
    for k in list(rk.keys()):
        if k.endswith('_randomization'):
            rk[k] = False
    v['randomization_kwargs'] = rk

    # 2) Env einfacher
    ek = dict(v['env_kwargs'])
    ek['goal_noise_range'] = [0.0, 0.0]
    ek['max_close_steps']  = 5
    ek['output_max'] = 0.05          # vorher 0.03
 
    ek['frame_stack_size'] = 1
    v['env_kwargs'] = ek

    # in randomization_kwargs sicherstellen:
    rk['albumentations_randomization'] = False
    v['randomization_kwargs'] = rk

    # Bild-Delay/Rauschen neutralisieren:
    ek['image_obs_noise_mean'] = 0.0
    ek['image_obs_noise_std']  = 0.0

    # 3) Längerer Horizont fürs Testen
    ak = dict(v['algorithm_kwargs'])
    ak['max_path_length'] = max(ak.get('max_path_length', 50), 200)
    v['algorithm_kwargs'] = ak

    # (Optional) Eval-Horizont angleichen
    ev = dict(v['eval_kwargs'])
    ev['max_path_length'] = max(ev.get('max_path_length', 50), 200)
    v['eval_kwargs'] = ev
    return v

def _load_variant():
    argv = sys.argv[:]
    sys.argv = [argv[0]]
    args = general_utils.argsparser()
    variant = general_utils.get_variant(args)
    sys.argv = argv
    return variant

def make_env(engine=None):
    variant = _load_variant()
    variant = apply_easy_mode(variant)
    env_kwargs = dict(variant['env_kwargs'])
    if engine:
        for k in ('engine','physics_engine','backend'):
            if k in ClothEnv.__init__.__code__.co_varnames:
                env_kwargs[k] = engine; break
    base = ClothEnv(**env_kwargs, randomization_kwargs=variant['randomization_kwargs'])
    return wrappers.NormalizedBoxEnv(base)

def scripted_action(obs, gain_xy=30.0, dz_down=-0.02, xy_thresh=0.01, z_hold=-0.04):
    ag = obs['achieved_goal'].reshape(-1, 3)[:6]
    dg = obs['desired_goal'].reshape(-1, 3)[:6]
    d  = (dg.mean(axis=0) - ag.mean(axis=0))
    ax, ay = d[0]*gain_xy, d[1]*gain_xy
    az = z_hold if np.hypot(d[0], d[1]) < xy_thresh else dz_down
    return np.clip([ax, ay, az], -1.0, 1.0).astype('float32')


def run(engine, episodes=3, max_path_length=100, gain_xy=20.0, dz_down=-0.005):
    env = make_env(engine)
    returns, successes = [], []
    for ep in range(episodes):
        obs = env.reset()
        ret, success = 0.0, False
        for t in range(max_path_length):
            a = scripted_action(obs, gain_xy=gain_xy, dz_down=dz_down)
            obs, r, done, info = env.step(a)
            ret += float(r)
            success = success or bool(info.get('is_success', False))
            if done: break
        returns.append(ret)
        successes.append(float(success))
    print(f"=== {engine} ===")
    print({"return_mean": float(np.mean(returns)),
           "success_rate": float(np.mean(successes))})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="mujoco")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-path-length", type=int, default=200)
    ap.add_argument("--gain-xy", type=float, default=30.0)
    ap.add_argument("--dz-down", type=float, default=-0.001)
    args, _ = ap.parse_known_args()
    run(args.engine, args.episodes, args.max_path_length, args.gain_xy, args.dz_down)
