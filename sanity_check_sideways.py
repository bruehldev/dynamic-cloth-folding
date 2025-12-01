import time
from pathlib import Path

import numpy as np
from rlkit.core import logger as rlkit_logger

from env.cloth_bullet.bullet_model_kwargs import (
    make_bullet_randomization_kwargs,
    make_env_kwargs,
)
from env.cloth_bullet.cloth_env_pybullet import ClothEnvBullet
from utils import general_utils
from utils.training_overrides import apply_training_env_overrides

DEMO_PATH = Path("./data/demos_pybullet.csv")


def _load_demo_actions(path: Path, output_max: float) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Demo file not found: {path}")

    demo_actions = np.genfromtxt(path, delimiter=",").astype(np.float32)
    if demo_actions.ndim == 1:
        demo_actions = np.expand_dims(demo_actions, axis=0)

    # Normalize by output_max to match the action space expected by the env
    normalized = np.clip(demo_actions / float(output_max), -1.0, 1.0)
    return normalized


def run_sideways_sanity_check():
    # Build a training-like variant (mirrors train.py) but force Bullet + viewer
    args = general_utils.argsparser()
    variant = general_utils.get_variant(args)
    variant["physics_backend"] = "bullet"

    variant["env_kwargs"]["has_viewer"] = True
    variant["env_kwargs"]["timestep"] = 1.0 / 480.0
    variant["randomization_kwargs"] = make_bullet_randomization_kwargs()
    variant["randomization_kwargs"]["render_size"] = [500, 500]
    variant["randomization_kwargs"]["task_name"] = "sideways"
    variant["env_kwargs"] = make_env_kwargs(variant.get("env_kwargs", {}))

    variant = apply_training_env_overrides(variant)

    # Apply test-friendly overrides AFTER defaults/overrides are merged
    variant["env_kwargs"]["output_max"] = 0.05
    variant["env_kwargs"]["success_distance"] = 0.187
    # variant["env_kwargs"]["fail_reward"] = -1.1
    # variant["env_kwargs"]["success_reward"] = 0.0

    print("Initialize Bullet cloth env for sideways folding...")
    env = ClothEnvBullet(
        **variant["env_kwargs"],
        randomization_kwargs=variant["randomization_kwargs"],
        logger=rlkit_logger,
    )

    obs = env.reset()
    print("\n--- STARTING SIDEWAYS DEMO ROLLOUT ---")
    print(
        f"Goal dim: {obs['desired_goal'].shape[0]}, Task: {variant['randomization_kwargs']['task_name']}"
    )

    demo_actions = _load_demo_actions(DEMO_PATH, variant["env_kwargs"]["output_max"])
    # Leave a few extra no-op steps after the demo to watch the cloth settle
    max_steps = len(demo_actions) + 30

    for i in range(max_steps):
        action = demo_actions[i] if i < len(demo_actions) else np.zeros(3, dtype=np.float32)
        obs, reward, done, info = env.step(action)

        dist = info.get("corner_distance", np.nan)
        corner_sum = info.get("corner_sum_error", np.nan)
        print(
            f"Step {i:03d} | Reward: {reward:.3f} | CornerDist: {dist:.3f} | SumErr: {corner_sum:.3f}"
        )

        time.sleep(0.05)

        if done and i >= len(demo_actions):
            print("Done signal received after demo rollout.")
            break

    env.close()


if __name__ == "__main__":
    run_sideways_sanity_check()
