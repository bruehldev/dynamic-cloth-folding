import time
from pathlib import Path

import cv2
import numpy as np

from env import cloth_env
from utils import general_utils
from utils.training_overrides import apply_training_env_overrides

DEMO_PATH = Path("./data/demos.csv")
IMAGE_DIR_NAME = "mujoco_sideways_images"


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
    variant["randomization_kwargs"]["task_name"] = "sideways_one_corner"

    # Ensure save folder exists because Mujoco env writes XML there
    Path(variant["save_folder"]).mkdir(parents=True, exist_ok=True)

    # Prepare image dump directories (mirrors capture_images structure)
    image_root = Path(variant["save_folder"]) / IMAGE_DIR_NAME
    corner_dir = image_root / "corners"
    eval_dir = image_root / "eval"
    cnn_color_full_dir = image_root / "cnn_color_full"
    cnn_color_dir = image_root / "cnn_color"
    cnn_gray_dir = image_root / "cnn_gray"
    for d in (corner_dir, eval_dir, cnn_color_full_dir, cnn_color_dir, cnn_gray_dir):
        d.mkdir(parents=True, exist_ok=True)

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

        # Save rendered images for debugging/visualization
        corner_img, eval_img, cnn_color_full_img, cnn_color_img, cnn_img = env.capture_images(
            info.get("corner_positions")
        )
        step_name = f"{i:03d}.png"
        cv2.imwrite(str(corner_dir / step_name), corner_img)
        cv2.imwrite(str(eval_dir / step_name), eval_img)
        cv2.imwrite(str(cnn_color_full_dir / step_name), cnn_color_full_img)
        cv2.imwrite(str(cnn_color_dir / step_name), cnn_color_img)
        cv2.imwrite(str(cnn_gray_dir / step_name), cnn_img)

        time.sleep(0.05)

        if done and i >= len(demo_actions):
            print("Done signal received after demo rollout.")
            break


if __name__ == "__main__":
    run_sideways_sanity_check_mujoco()
