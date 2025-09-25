# tools/demo_policy_success.py
#!/usr/bin/env python3
import numpy as np, argparse, sys, os
from utils import general_utils
from rlkit.envs import wrappers
from env.cloth_env import ClothEnv

def _load_variant():
    argv = sys.argv[:]
    sys.argv = [argv[0]]
    args = general_utils.argsparser()
    v = general_utils.get_variant(args)
    sys.argv = argv
    return v

def make_env():
    v = _load_variant()
    # make it easy but honest
    rk = dict(v['randomization_kwargs'])
    for k in list(rk.keys()):
        if k.endswith('_randomization'):
            rk[k] = False
    v['randomization_kwargs'] = rk

    ek = dict(v['env_kwargs'])
    ek['goal_noise_range'] = [0.0, 0.0]
    ek['success_distance'] = 0.05
    ek['frame_stack_size'] = 1
    v['env_kwargs'] = ek

    base = ClothEnv(**v['env_kwargs'], randomization_kwargs=v['randomization_kwargs'])
    return wrappers.NormalizedBoxEnv(base)

def load_demo(path, output_max):
    demo = np.loadtxt(path, delimiter=",")  # shape (T, 3) in meters
    # convert meters -> raw action in [-1,1]
    raw = demo / float(output_max)
    return np.clip(raw.astype(np.float32), -1.0, 1.0)

def centroid_delta(obs):
    ag = obs["achieved_goal"].reshape(-1, 3)[:6].mean(axis=0)
    dg = obs["desired_goal"].reshape(-1, 3)[:6].mean(axis=0)
    return dg - ag  # (dx, dy, dz) in meters (goal space)

def run(episodes, max_path_length, demo_path):
    env = make_env()
    outmax = getattr(env.unwrapped, "output_max", 0.03)
    demo_actions = load_demo(demo_path, outmax)  # normalized actions
    T = len(demo_actions)

    rets, succ = [], []
    for ep in range(episodes):
        obs = env.reset()
        ret, success = 0.0, False
        for t in range(max_path_length):
            if t < T:
                a = demo_actions[t]
            else:
                # after demo ends: gentle XY towards goal, hold Z slightly down
                d = centroid_delta(obs)
                ax = np.clip( d[0] * 30.0, -1, 1)
                ay = np.clip( d[1] * 30.0, -1, 1)
                az = -0.02  # light contact
                a = np.array([ax, ay, az], dtype=np.float32)

            obs, r, done, info = env.step(a)
            ret += float(r)
            success = success or bool(info.get('is_success', False))
            if done: break

        rets.append(ret); succ.append(float(success))

    print("=== demo_policy ===")
    print({"return_mean": float(np.mean(rets)),
           "success_rate": float(np.mean(succ))})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-path-length", type=int, default=200)
    ap.add_argument("--demo", default="data/demos.csv")
    args, _ = ap.parse_known_args()
    run(args.episodes, args.max_path_length, args.demo)
