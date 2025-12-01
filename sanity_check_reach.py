import time

import numpy as np
from rlkit.core import logger as rlkit_logger

from env.cloth_bullet.bullet_model_kwargs import make_bullet_randomization_kwargs, make_env_kwargs
from env.cloth_bullet.cloth_env_pybullet import ClothEnvBullet
from utils import general_utils
from utils.training_overrides import apply_training_env_overrides


def run_sanity_check():
    # 1. Setup Config - Mimic train.py logic
    args = general_utils.argsparser()
    variant = general_utils.get_variant(args)

    # Force backend to bullet for this script
    variant["physics_backend"] = "bullet"

    # Bullet specific setup
    # Force viewer for sanity check
    variant["env_kwargs"]["has_viewer"] = True
    variant["randomization_kwargs"] = make_bullet_randomization_kwargs()
    variant["env_kwargs"] = make_env_kwargs(variant.get("env_kwargs", {}))

    variant = apply_training_env_overrides(variant)

    # Apply Sanity Check specific overrides
    variant["env_kwargs"]["timestep"] = 1.0 / 480.0
    variant["randomization_kwargs"]["render_size"] = [500, 500]

    # IMPORTANT: Set a goal that requires moving DOWN (negative Z) to test your fix
    variant["randomization_kwargs"]["simple_ee_goal_I"] = [-0.1, -0.1, 0.1]

    print("Initialize Env...")
    env = ClothEnvBullet(
        **variant["env_kwargs"],
        randomization_kwargs=variant["randomization_kwargs"],
        logger=rlkit_logger,
    )

    obs = env.reset()

    print("\n--- STARTING CONTROL LOOP ---")
    print(f"Goal (Inertial): {env.goal}")

    for i in range(100):
        # 2. CHEAT: Calculate the perfect action manually
        # Action = Vector from EE to Goal, normalized
        ee_pos_I = obs["achieved_goal"]  # This is EE position in Inertial frame
        desired_goal = obs["desired_goal"]

        direction = desired_goal - ee_pos_I
        dist = np.linalg.norm(direction)

        if dist < 0.01:
            print(f"Step {i}: TARGET REACHED! Dist: {dist:.4f}")
            action = np.zeros(3)  # Stay there
        else:
            # Normalize and scale to max speed
            action = (direction / dist) * 1.0

        # 3. Step the environment
        obs, reward, done, info = env.step(action)

        # 4. Debug Print
        ee_z = obs["achieved_goal"][2]
        print(
            f"Step {i:03d} | Reward: {reward:.3f} | Dist: {info['dist_to_target']:.3f} | EE_Z (Inertial): {ee_z:.3f}"
        )

        time.sleep(0.05)  # Slow down so you can watch

        if done:
            print("Done signal received!")
            break

    env.close()


if __name__ == "__main__":
    run_sanity_check()
