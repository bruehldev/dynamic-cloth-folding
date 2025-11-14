import json
import os
from typing import Any, Dict

import gym
import numpy as np


class _NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if hasattr(obj, "dtype"):
            try:
                return obj.item()
            except Exception:
                return str(obj)
        return json.JSONEncoder.default(self, obj)


def _stats(arr: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(arr)
    if arr.size == 0:
        return {"shape": list(arr.shape), "sum": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "shape": list(arr.shape),
        "sum": float(arr.sum()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


class BulletParityLogger(gym.Env):
    """
    Thin wrapper that logs reset snapshots and per-step rows to JSON files.
    Never raises if logging fails; swallows exceptions to keep training running.
    """

    metadata = {}

    def __init__(self, env: gym.Env, log_dir: str):
        super().__init__()
        os.makedirs(log_dir, exist_ok=True)
        self.env = env
        self.log_dir = log_dir
        self.reset_path = os.path.join(log_dir, "reset_bullet.json")
        self.steps_path = os.path.join(log_dir, "steps_bullet.jsonl")
        self._steps_fp = open(self.steps_path, "a", buffering=1)
        self.t = 0

        # Bubble up spaces
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    # --------------- gym.Env interface ---------------
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        try:
            snap = self._build_reset_snapshot(obs)
            with open(self.reset_path, "w") as f:
                json.dump(snap, f, cls=_NumpyJSONEncoder, indent=2)
        except Exception:
            pass
        self.t = 0
        return obs

    def step(self, action):
        raw_action = np.asarray(action, dtype=np.float32).copy()
        obs, reward, done, info = self.env.step(raw_action)
        try:
            row = self._build_step_row(raw_action, obs, reward, done, info)
            self._steps_fp.write(json.dumps(row, cls=_NumpyJSONEncoder) + "\n")
        except Exception:
            pass
        self.t += 1
        return obs, reward, done, info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        try:
            self._steps_fp.close()
        except Exception:
            pass
        return self.env.close()

    def __getattr__(self, name):
        # Delegate everything else to the wrapped env
        return getattr(self.env, name)

    # --------------- helpers ---------------
    def _build_reset_snapshot(self, obs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        e = self.env  # type: ignore
        cam = getattr(e, "camera", None)
        world = getattr(e, "world", None)
        robot = getattr(e, "robot", None)
        cloth = getattr(e, "cloth", None)
        task = getattr(e, "task", None)
        # Bullet stores this dict as `env.kwargs`; MuJoCo envs often use `env.randomization_kwargs`.
        cfg = getattr(e, "kwargs", None) or getattr(e, "randomization_kwargs", {})  # <- key fix

        # Camera (episode) params if available
        cam_snap = {}
        if cam is not None:
            cam_snap = {
                "render_size": list(
                    getattr(cam, "render_size", cfg.get("render_size", (None, None)))
                ),
                "episode": {
                    "center": getattr(cam, "_episode_center", None),
                    "eye": getattr(cam, "_episode_eye", None),
                    "up": getattr(cam, "_episode_up", None),
                    "fov_deg": getattr(cam, "_episode_fov", None),
                },
                "config": cfg.get("camera_config", {}),
            }

        # World / physics params
        world_snap = {}
        if world is not None:
            world_snap = {
                "timestep": float(getattr(world, "timestep", getattr(e, "timestep", 0.0))),
                "gravity_vec": list(getattr(world, "gravity_vec", [0, 0, -9.81])),
                "table_top_z": float(getattr(world, "_table_z", 0.0)),
                "table_dyn": {
                    "lateralFriction": float(getattr(world, "table_lateral_friction", 0.0)),
                    "rollingFriction": float(getattr(world, "table_rolling_friction", 0.0)),
                    "spinningFriction": float(getattr(world, "table_spinning_friction", 0.0)),
                    "restitution": float(getattr(world, "table_restitution", 0.0)),
                },
            }

        # Robot snapshot
        robot_snap = {}
        if robot is not None:
            robot_snap = {
                "ee_link_index": int(getattr(robot, "ee_link_index", -1)),
                "ee_W": list(
                    self._safe_call(robot, "get_ee_position_W", default=[None, None, None])
                ),
                "ik": {
                    "use_orientation": bool(getattr(robot, "ik_use_orientation", True)),
                },
            }

        # Cloth snapshot
        cloth_snap = {}
        if cloth is not None:
            cloth_snap = {
                "scale": float(getattr(cloth, "scale", 0.0)),
                "mass": float(getattr(cloth, "mass", 0.0)),
                "spring_k": float(getattr(cloth, "springElasticStiffness", 0.0)),
                "spring_c": float(getattr(cloth, "springDampingStiffness", 0.0)),
                "friction": float(getattr(cloth, "frictionCoeff", 0.0)),
                "thickness": float(getattr(cloth, "thickness", 0.0)),
            }

        # Task snapshot
        task_snap = {}
        if task is not None:
            task_snap = {
                "name": getattr(e, "task_name", None),
                "success_distance": float(getattr(task, "success_distance", 0.0)),
                # we'll overwrite constraints below with enriched entries
                "constraints": [],
            }

            # --- Enrich constraints with origin/target positions in W and I frames ---
            try:
                verts_W = e.cloth.get_positions_W()  # key -> np.array([x,y,z])
                sites_map = getattr(e.cloth, "sites", {})  # "S{r}_{c}" -> key into verts_W
                origin_W = np.array(getattr(e, "relative_origin", [0, 0, 0]), dtype=float)
                for c in getattr(task, "constraints", []):
                    c_out = dict(c)  # shallow copy of the original constraint dict
                    ok, tk = c.get("origin"), c.get("target")
                    v_ok = sites_map.get(ok, None)
                    v_tk = sites_map.get(tk, None)
                    oW = np.array(verts_W[v_ok], dtype=float) if v_ok in verts_W else None
                    tW = np.array(verts_W[v_tk], dtype=float) if v_tk in verts_W else None
                    # I-frame is world minus episode origin (axes aligned with world)
                    oI = (oW - origin_W) if oW is not None else None
                    tI = (tW - origin_W) if tW is not None else None
                    if oW is not None:
                        c_out["origin_W"] = oW.tolist()
                    if tW is not None:
                        c_out["target_W"] = tW.tolist()
                    if oI is not None:
                        c_out["origin_I"] = oI.tolist()
                    if tI is not None:
                        c_out["target_I"] = tI.tolist()
                    task_snap["constraints"].append(c_out)
            except Exception:
                # if anything goes wrong, fall back to the raw constraints
                task_snap["constraints"] = getattr(task, "constraints", [])

        # Workspace bounds
        ws = {
            "origin_W": list(getattr(e, "relative_origin", [None, None, None])),
            "limits_min": list(getattr(e, "limits_min", [0, 0, 0])),
            "limits_max": list(getattr(e, "limits_max", [0, 0, 0])),
            "abs_min_W": list(getattr(e, "min_absolute_W", [None, None, None])),
            "abs_max_W": list(getattr(e, "max_absolute_W", [None, None, None])),
        }

        # Observation stats (keep light; images are summarized)
        obs = obs or {}
        obs_stats = {
            "observation": _stats(obs.get("observation", np.array([]))),
            "robot_observation": _stats(obs.get("robot_observation", np.array([]))),
            "achieved_goal": _stats(obs.get("achieved_goal", np.array([]))),
            "desired_goal": _stats(obs.get("desired_goal", np.array([]))),
            "image": _stats(obs.get("image", np.array([]))),
        }

        return {
            "backend": "bullet",
            "episode": {"t0": int(self.t)},
            "world": world_snap,
            "camera": cam_snap,
            "workspace": ws,
            "robot": robot_snap,
            "cloth": cloth_snap,
            "task": task_snap,
            "obs0_stats": obs_stats,
        }

    def _build_step_row(self, raw_action, obs, reward, done, info) -> Dict[str, Any]:
        e = self.env  # type: ignore
        distances = {k: info.get(f"corner_{k}") for k in ("0", "1", "2", "3")}
        row = {
            "t": int(self.t),
            "raw_action": raw_action,
            "ee_W": list(self._safe_call(e.robot, "get_ee_position_W"))
            if hasattr(e, "robot")
            else None,
            "ee_target_W": list(getattr(e, "desired_pos_ctrl_W", [])),
            "ee_target_step_W": list(getattr(e, "desired_pos_step_W", [])),
            "reward": float(reward),
            "done": bool(done),
            "is_success": bool(info.get("is_success", False)),
            "distances": distances,
            "corner_sum_error": info.get("corner_sum_error"),
            "ctrl_error": info.get("ctrl_error"),
            "delta_size": info.get("delta_size"),
        }
        return row

    @staticmethod
    def _safe_call(obj, method, default=None):
        try:
            return getattr(obj, method)()
        except Exception:
            return default
