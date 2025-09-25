#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
from utils import general_utils
from env.cloth_env import ClothEnv
from rlkit.envs import wrappers

def main():
    # load variant like train.py
    argv = sys.argv[:]; sys.argv = [argv[0]]
    args = general_utils.argsparser()
    variant = general_utils.get_variant(args)
    sys.argv = argv

    envkw = dict(variant['env_kwargs'])
    base = ClothEnv(**envkw, randomization_kwargs=variant['randomization_kwargs'])
    env = wrappers.NormalizedBoxEnv(base)

    inv = {
        "backend": "mujoco",
        "timestep": float(base.timestep),
        "control_frequency": int(base.control_frequency),
        "substeps": int(base.substeps),
        "output_max": float(base.output_max),
        "ctrl_filter": float(base.filter) if base.filter is not None else None,
        "kp": float(base.kp),
        "damping_ratio": float(base.damping_ratio),
        "limits_min": [float(x) for x in base.limits_min],
        "limits_max": [float(x) for x in base.limits_max],
        # derived:
        "sim_time_per_action": float(base.timestep * base.substeps),
    }

    out_dir = os.path.join(variant["save_folder"], "spec_dump")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(out_dir, "control_invariants_mujoco.json")
    with open(out_path, "w") as f: json.dump(inv, f, indent=2)
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
