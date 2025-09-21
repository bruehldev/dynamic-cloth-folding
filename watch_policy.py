# watch_policy.py
import os, time, glob, copy, argparse
import numpy as np
import torch
import cv2
import sys

# safe GL defaults (headless render → show via OpenCV)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.pop("LD_PRELOAD", None)
os.environ.setdefault("DISPLAY", ":0")

from utils import general_utils
from rlkit.envs import wrappers
from rlkit.torch import pytorch_util
from rlkit.torch.sac import policies as sac_policies
from rlkit.torch import networks
from env import cloth_env

# ---- your watch argparse ----
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", type=str, default=None)
ap.add_argument("--episodes", type=int, default=10)
ap.add_argument("--slow", type=float, default=0.01)
ap.add_argument("--color", action="store_true")
watch_args = ap.parse_args()

# ---- isolate project arg parser from our flags ----
argv_backup = sys.argv[:]          # save user args (--episodes, etc.)
sys.argv = [sys.argv[0]]           # hide them from general_utils.argsparser()
v_args = general_utils.argsparser()
sys.argv = argv_backup             # restore

variant = general_utils.get_variant(v_args)

def find_latest_policy(save_folder, user_ckpt=None):
    if user_ckpt:
        return user_ckpt
    # try typical rlkit filenames
    patterns = [
        "policy*epoch*.pt", "policy*epoch*.pth",
        "policy*.pt", "policy*.pth",
        "*policy*.pt", "*policy*.pth",
        "snapshot_epoch_*", "params.pkl",
    ]
    cands = []
    for pat in patterns:
        cands += glob.glob(os.path.join(save_folder, pat))
    if not cands:
        raise FileNotFoundError(f"No policy checkpoint found in {save_folder}")
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]

def build_policy(env_dims, variant):
    # value nets are not needed for inference; just build the policy with correct sizes
    policy = sac_policies.TanhScriptPolicy(
        output_size=env_dims['action_dim'],
        added_fc_input_size=env_dims['added_fc_input_size'],
        **variant['policy_kwargs'],
    )
    return policy

def load_policy(policy, path, map_location):
    obj = torch.load(path, map_location=map_location)
    # Handle a few common rlkit save formats
    if isinstance(obj, dict):
        if "policy" in obj and hasattr(obj["policy"], "state_dict"):
            policy.load_state_dict(obj["policy"].state_dict())
        elif "policy_state_dict" in obj:
            policy.load_state_dict(obj["policy_state_dict"])
        else:
            # maybe it's already a state_dict?
            try:
                policy.load_state_dict(obj)
            except Exception as e:
                raise RuntimeError(f"Unrecognized checkpoint format at {path}: {e}")
    else:
        # direct state_dict
        policy.load_state_dict(obj)
    policy.eval()
    return policy

def make_obs_processor(env_keys, variant):
    add_keys = variant['path_collector_kwargs'].get('additional_keys', [])
    if isinstance(add_keys, str):
        try:
            add_keys = eval(add_keys)
        except Exception:
            add_keys = []

    def _proc(o):
        obs = o[env_keys['path_collector_observation_key']]
        for k in add_keys:
            obs = np.hstack((obs, o[k]))
        return np.hstack((obs, o[env_keys['desired_goal_key']]))
    return _proc

def _policy_action(pi, obs):
    # Ensure proper dtype/shape
    obs = np.asarray(obs, dtype=np.float32)
    # Call policy and accept either action or (action, info)
    out = pi.get_action(obs)
    return out[0] if isinstance(out, tuple) else out

def main():
    # args = ap.parse_args()  # REMOVE this line

    # build a single env with offscreen rendering (EGL) for display
    env = cloth_env.ClothEnv(
        **variant['env_kwargs'],
        randomization_kwargs=variant['randomization_kwargs'],
        has_viewer=True,
        viewer_mode="offscreen",
    )
    env = general_utils.get_randomized_env(
        wrappers.NormalizedBoxEnv(env),
        randomization_kwargs=variant['randomization_kwargs'],
    )

    # key/dim info used by the policy
    env_keys, env_dims = general_utils.get_keys_and_dims(variant, env)

    # policy
    policy = build_policy(env_dims, variant)
    ckpt = find_latest_policy(variant["save_folder"], watch_args.ckpt)
    policy = load_policy(policy, ckpt, map_location=pytorch_util.device)
    eval_policy = sac_policies.MakeDeterministic(policy)

    obs_proc = make_obs_processor(env_keys, variant)

    # OpenCV window
    cv2.namedWindow("MuJoCo Policy", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("MuJoCo Policy", 700, 700)

    max_len = variant['eval_kwargs']['max_path_length']
    print(f"Running {watch_args.episodes} episodes with checkpoint:\n  {ckpt}")

    for ep in range(watch_args.episodes):
        o = env.reset()
        ep_ret = 0.0
        for t in range(max_len):
            in_obs = obs_proc(o)
            with torch.no_grad():
                a = _policy_action(eval_policy, in_obs)
            o, r, d, info = env.step(a)
            ep_ret += float(r)

            # frame to display
            if watch_args.color:
                # Show color/overlay view from env (corners in red)
                _, eval_img, full_color, color_100, gray_100 = env.capture_images(aux_output=None, mask_type="corners")
                frame = color_100[..., ::-1]  # BGR->RGB if you prefer
            else:
                img = env.get_image_obs().reshape(env.image_size[1], env.image_size[0])
                frame = (img * 255).astype("uint8")

            cv2.imshow("MuJoCo Policy", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                d = True  # ESC to end early
            if d:
                break
            if watch_args.slow > 0:
                time.sleep(watch_args.slow)
        print(f"Episode {ep+1}: return={ep_ret:.3f}, success={info.get('is_success')}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
