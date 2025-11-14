# mujoco_parity_logger.py
# Usage:
#   from cloth_env import ClothEnv
#   from mujoco_parity_logger import MuJoCoParityLogger
#
#   env = ClothEnv(...your kwargs...)
#   logger = MuJoCoParityLogger(env, log_dir="logs/mj_run_001")
#   obs = logger.reset()          # writes reset.json
#   for t in range(200):
#       a = np.random.uniform(-1, 1, size=3).astype(np.float32)
#       obs, rew, done, info = logger.step(a)  # appends to steps.jsonl
#       if done: break
#   logger.close()

import json
import os
import time
from typing import Any, Dict, List

import numpy as np


class _NumpyJSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        return super().default(o)


def _to_list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return x


def _safe(obj, keys: List[str]):
    out = {}
    for k in keys:
        if hasattr(obj, k):
            v = getattr(obj, k)
            out[k] = _to_list(v)
    return out


class MuJoCoParityLogger:
    """
    Thin wrapper that logs a 'reset snapshot' and one JSON line per step.

    Files written in `log_dir`:
      - reset.json     : scene + physics + camera + goal snapshot
      - steps.jsonl    : one JSON object per env.step()
    """

    def __init__(self, env, log_dir: str):
        self.env = env
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.reset_path = os.path.join(self.log_dir, "reset_mujoco.json")
        self.steps_path = os.path.join(self.log_dir, "steps_mujoco.jsonl")
        self._steps_fp = open(self.steps_path, "a", buffering=1)  # line-buffered
        self.t = 0

    # --- proxy attributes so this wrapper behaves like the underlying gym.Env ---
    def __getattr__(self, name):
        # Allow access to observation_space, action_space, sim, etc.
        # (Only called if the attribute wasn't found on self)
        return getattr(self.env, name)

    # Optional: ensure context-manager friendliness if used with "with"
    __enter__ = lambda self: self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # --------- public API ----------
    def reset(self):
        obs = self.env.reset()
        try:
            snap = self._build_reset_snapshot(obs)
            with open(self.reset_path, "w") as f:
                json.dump(snap, f, cls=_NumpyJSONEncoder, indent=2)
        except Exception:
            # swallow logging errors; never break env.reset()
            pass
        self.t = 0
        return obs

    def step(self, action: np.ndarray):
        # Log the *input* action as raw_action (your env scales internally)
        raw_action = np.asarray(action, dtype=np.float32).copy()

        obs, reward, done, info = self.env.step(raw_action)
        try:
            row = self._build_step_row(raw_action, obs, reward, done, info)
            self._steps_fp.write(json.dumps(row, cls=_NumpyJSONEncoder) + "\n")
        except Exception:
            # don’t interrupt training if logging hiccups
            pass
        self.t += 1
        return obs, reward, done, info

    def close(self):
        try:
            self._steps_fp.close()
        except Exception:
            pass

    # --------- reset snapshot ----------
    def _build_reset_snapshot(self, obs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        env = self.env
        sim = env.sim

        # Physics / timing
        physics = {
            "timestep": float(env.timestep),
            "control_frequency": float(env.control_frequency),
            "substeps": int(env.substeps),
            "between_steps_ms": float(env.between_steps),
            "ctrl_filter": float(env.filter),
            "output_max": float(env.output_max),
        }

        # World
        gravity = _to_list(sim.model.opt.gravity) if hasattr(sim.model.opt, "gravity") else None
        world = {
            "gravity": gravity,
        }

        # Workspace
        # min/max are set during update_osc_values() after reset in your env
        workspace = {
            "relative_origin_W": _to_list(env.relative_origin),
            "min_absolute_W": _to_list(getattr(env, "min_absolute_W", None)),
            "max_absolute_W": _to_list(getattr(env, "max_absolute_W", None)),
        }

        # Camera
        cam_name = getattr(env, "train_camera", "train_camera")
        cam_id = sim.model.camera_name2id(cam_name)

        # fovy is widely available on mujoco_py models, keep as-is if present
        try:
            fovy = float(sim.model.cam_fovy[cam_id])
        except Exception:
            fovy = None

        # cam_near/cam_far are NOT always exposed on mujoco_py builds.
        # Try camera arrays first; if absent, fall back to global vis.map znear/zfar.
        near = None
        far = None
        try:
            if hasattr(sim.model, "cam_near"):
                near = float(sim.model.cam_near[cam_id])
            if hasattr(sim.model, "cam_far"):
                far = float(sim.model.cam_far[cam_id])
        except Exception:
            pass
        if near is None or far is None:
            try:
                # global znear/zfar (works on older mujoco_py)
                znear = getattr(sim.model.vis.map, "znear", None)
                zfar = getattr(sim.model.vis.map, "zfar", None)
                near = float(znear) if znear is not None else near
                far = float(zfar) if zfar is not None else far
            except Exception:
                pass

        render_w = int(env.randomization_kwargs["camera_config"]["width"])
        render_h = int(env.randomization_kwargs["camera_config"]["height"])
        image_w, image_h = env.image_size

        camera_matrix, camera_T = env.get_camera_matrices(cam_name, image_w, image_h)
        camera = {
            "type": cam_name,
            "fovy_deg": fovy,
            "near": near,
            "far": far,
            "render_size": [int(render_w), int(render_h)],
            "image_size": [int(image_w), int(image_h)],
            "camera_matrix": _to_list(camera_matrix),
            "camera_transform": _to_list(camera_T),
        }

        # Goal & constraints
        goal = {
            "desired_goal": _to_list(env.goal),
            "success_distance": float(env.success_distance),
            "max_close_steps": int(env.max_close_steps),
            "goal_noise": float(getattr(env, "goal_noise", 0.0)),
        }

        constraints = []
        try:
            # distance at reset for each origin→target pair
            for c in env.constraints:
                origin = c["origin"]
                target = c["target"]
                o = sim.data.get_site_xpos(origin).copy() - env.relative_origin
                t = sim.data.get_site_xpos(target).copy() - env.relative_origin
                constraints.append(
                    {
                        "origin": origin,
                        "target": target,
                        "origin_I": _to_list(o),
                        "target_I": _to_list(t),
                        "distance": float(np.linalg.norm(o - t)),
                    }
                )
        except Exception:
            pass

        # Cloth measurements (reset-time)
        cloth = self._measure_cloth(env)

        # Corner UVs at reset (normalized 0..1 in the cropped image)
        try:
            corner_uv = env.post_action_image_capture()
            corner_uv = _to_list(corner_uv)
        except Exception:
            corner_uv = None

        snapshot = {
            "backend": "mujoco",
            "timestamp": time.time(),
            "physics": physics,
            "world": world,
            "workspace": workspace,
            "camera": camera,
            "goal": goal,
            "constraints": constraints,
            "cloth": cloth,
            "corner_uv_at_reset": corner_uv,
            # raw numeric params that your env includes when DR is on
            "mujoco_model_numerical_values": _to_list(
                getattr(env, "mujoco_model_numerical_values", [])
            ),
        }
        return snapshot

    def _measure_cloth(self, env) -> Dict[str, Any]:
        """Estimate cloth XY span from edge sites and report the four corners in world coords."""
        try:
            edges = env.get_cloth_edge_positions_W()
            pts = np.array([p for p in edges.values()])
            min_xy = pts[:, :2].min(axis=0)
            max_xy = pts[:, :2].max(axis=0)
            aabb_xy = (max_xy - min_xy).tolist()

            # fetch nominal corners if available
            names = ["S0_8", "S8_8", "S0_0", "S8_0"]
            corners = {}
            for n in names:
                if n in edges:
                    corners[n] = _to_list(edges[n])
            return {"edge_aabb_xy": aabb_xy, "corner_W": corners}
        except Exception:
            return {}

    # --------- per-step row ----------
    def _build_step_row(
        self,
        raw_action: np.ndarray,
        obs: Dict[str, np.ndarray],
        reward: float,
        done: bool,
        info: Dict[str, Any],
    ) -> Dict[str, Any]:
        env = self.env

        pos_W = env.get_ee_position_W()
        desired_step_W = getattr(env, "desired_pos_step_W", None)
        desired_ctrl_W = getattr(env, "desired_pos_ctrl_W", None)

        if desired_ctrl_W is not None:
            ctrl_error = float(np.linalg.norm(np.asarray(desired_ctrl_W) - np.asarray(pos_W)))
        else:
            ctrl_error = None

        # constraint distances (robust even if info lacks them)
        try:
            dists = env.get_corner_constraint_distances()
            corner_fields = {
                "corner_0": float(dists.get("0", np.nan)),
                "corner_1": float(dists.get("1", np.nan)),
                "corner_2": float(dists.get("2", np.nan)),
                "corner_3": float(dists.get("3", np.nan)),
                "corner_sum_error": float(sum(dists.values())),
            }
        except Exception:
            corner_fields = {
                "corner_0": info.get("corner_0"),
                "corner_1": info.get("corner_1"),
                "corner_2": info.get("corner_2"),
                "corner_3": info.get("corner_3"),
                "corner_sum_error": info.get("corner_sum_error"),
            }

        row = {
            "t": int(self.t),
            "raw_action": _to_list(raw_action),
            "delta_size": float(np.linalg.norm(raw_action)),
            "desired_pos_step_W": _to_list(desired_step_W),
            "desired_pos_ctrl_W": _to_list(desired_ctrl_W),
            "ee_W": _to_list(pos_W),
            "ctrl_error": ctrl_error,
            # achieved/desired goals are already in I-frame in your env
            "achieved_goal": _to_list(obs.get("achieved_goal")),
            "desired_goal": _to_list(obs.get("desired_goal")),
            "corner_positions": _to_list(info.get("corner_positions")),
            "reward": float(reward),
            "is_success": bool(
                info.get("is_success", reward > getattr(self.env, "fail_reward", -1e9))
            ),
            "done": bool(done),
            # optional extras if present
            "env_memory_usage": info.get("env_memory_usage"),
        }
        row.update(corner_fields)
        return row
