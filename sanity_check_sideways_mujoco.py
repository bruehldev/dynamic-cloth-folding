import time
from pathlib import Path

import numpy as np

from env import cloth_env
from utils import general_utils
from utils.training_overrides import apply_training_env_overrides

DEMO_PATH = Path("./data/demos.csv")


def _load_demo_actions(path: Path, output_max: float) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Demo file not found: {path}")

    demo_actions = np.genfromtxt(path, delimiter=",").astype(np.float32)
    if demo_actions.ndim == 1:
        demo_actions = np.expand_dims(demo_actions, axis=0)

    # Normalize by output_max to match the env's action space
    normalized = np.clip(demo_actions / float(output_max), -1.0, 1.0)
    return normalized


def run_sideways_sanity_check_mujoco():
    # Build a training-like variant but force Mujoco settings
    args = general_utils.argsparser()
    variant = general_utils.get_variant(args)
    variant["physics_backend"] = "mujoco"

    # Viewer on for debugging and a slightly finer timestep for smoother playback
    variant["env_kwargs"]["has_viewer"] = True
    variant["env_kwargs"]["timestep"] = variant["env_kwargs"].get("timestep", 0.01)

    variant = apply_training_env_overrides(variant)

    # Ensure save folder exists because Mujoco env writes XML there
    Path(variant["save_folder"]).mkdir(parents=True, exist_ok=True)

    print("Initialize Mujoco cloth env for sideways folding...")
    env = cloth_env.ClothEnv(
        **variant["env_kwargs"],
        randomization_kwargs=variant["randomization_kwargs"],
    )

    obs = env.reset()
    print("\n--- STARTING SIDEWAYS DEMO ROLLOUT (MuJoCo) ---")
    print(f"Desired goal dim: {obs['desired_goal'].shape[0]}")

    demo_actions = _load_demo_actions(DEMO_PATH, variant["env_kwargs"]["output_max"])
    max_steps = len(demo_actions) + 30  # allow settling time after the demo

    for i in range(max_steps):
        action = demo_actions[i] if i < len(demo_actions) else np.zeros(3, dtype=np.float32)
        obs, reward, done, info = env.step(action)

        c1 = info.get("corner_1", np.nan)
        csum = info.get("corner_sum_error", np.nan)
        print(f"Step {i:03d} | Reward: {reward:.3f} | Corner1: {c1:.3f} | SumErr: {csum:.3f}")

        time.sleep(0.05)

        if done and i >= len(demo_actions):
            print("Done signal received after demo rollout.")
            break


if __name__ == "__main__":
    run_sideways_sanity_check_mujoco()
