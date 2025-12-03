import os
from collections import deque
from multiprocessing import current_process
from typing import Any, Optional

import cv2
import gym
import numpy as np
import psutil
import pybullet as p
from gym.utils import EzPickle, seeding

from env.cloth_bullet import debug_util  # noqa: F401
from env.cloth_bullet.camera import Camera
from env.cloth_bullet.deformable_cloth import DeformableCloth
from env.cloth_bullet.folding_task import FoldingTask
from env.cloth_bullet.panda_robot import PandaRobot
from env.cloth_bullet.pybullet_world import PyBulletWorld


class _NoOpLogger:
    """A logger that does nothing, useful for environments in worker processes."""

    def log(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        """Return a no-op function for any attribute access."""
        return self.log


class BulletClothEnv_:
    """
    PyBullet-Port with identical external API to ClothEnv (MuJoCo).
    Coordinates the different components of the simulation.
    """

    def __init__(
        self,
        randomization_kwargs,
        # --- Env kwargs ---
        control_frequency,
        ctrl_filter,
        damping_ratio,
        frame_stack_size,
        image_obs_noise_mean,
        image_obs_noise_std,
        kp,
        max_close_steps,
        model_kwargs_path,
        output_max,
        robot_observation,
        timestep,
        # -> Task settings
        fail_reward,
        extra_reward,
        goal_noise,
        goal_noise_range,
        sparse_dense,
        success_distance,
        success_reward,
        # --- Other ---
        save_folder,
        has_viewer=False,
        logger: Optional[Any] = None,
    ):
        self.logger = logger if logger is not None else _NoOpLogger()

        if current_process().name != "MainProcess":
            has_viewer = False
            self.logger.log("Viewer disabled in subprocess.")

        self.process = psutil.Process(os.getpid())
        self.seed()

        # --- Init Params from kwargs ---
        self.randomization_kwargs = randomization_kwargs

        # --- Env settings ---
        self.task_name = self.randomization_kwargs["task_name"]
        self.control_frequency = control_frequency
        self.ctrl_filter = ctrl_filter
        self.save_folder = save_folder
        self.timestep = timestep
        self.substeps = max(1, int(1.0 / (self.timestep * self.control_frequency)))
        steps_per_second = 1.0 / self.timestep
        self.between_steps = max(1, int(1000.0 / steps_per_second))
        self.output_max = output_max
        self.robot_observation = robot_observation
        self.max_close_steps = max_close_steps
        self.frame_stack_size = frame_stack_size

        # --- Task settings ---
        self.fail_reward = fail_reward
        self.extra_reward = extra_reward
        self.goal_noise = goal_noise
        self.goal_noise_range = goal_noise_range
        self.success_distance = success_distance
        self.sparse_dense = sparse_dense
        self.success_reward = success_reward

        self.image_size = (
            self.randomization_kwargs["image_size"],
            self.randomization_kwargs["image_size"],
        )

        # --- Optional simple EE-reaching task ---
        self.simple_ee_task = False
        self.simple_ee_goal_I = np.array(
            self.randomization_kwargs.get("simple_ee_goal_I", [-0.1, -0.1, 0.1]), dtype=np.float32
        )

        self.has_viewer = has_viewer
        self.image_obs_noise_mean = image_obs_noise_mean
        self.image_obs_noise_std = image_obs_noise_std

        # Define action space before reset is called
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        self.world = PyBulletWorld(self.has_viewer, self.timestep, self.randomization_kwargs)
        self.camera = Camera(self.image_size, self.randomization_kwargs)
        self.frame_stack = deque([], maxlen=self.frame_stack_size)
        self._ws_vis_id = None  # Debug: id of the translucent workspace box
        self._ws_line_ids = []  # Debug: store wireframe line ids so we can clear them
        self._task_line_ids = []  # Debug: debug lines (origin→target / origin→goal)
        self._task_marker_ids = []  # Debug: tiny spheres & labels for sites/goals
        self._simple_goal_debug_ids = []  # Debug: markers for the simple EE goal

        robot_cfg = self.randomization_kwargs["robot"]
        self.limits_min = robot_cfg["workspace_limits_min"]
        self.limits_max = robot_cfg["workspace_limits_max"]

        self.reset()

        # --- Define observation space ---
        obs = self.get_obs()
        self.observation_space = gym.spaces.Dict(
            dict(
                desired_goal=gym.spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype=np.float32
                ),
                achieved_goal=gym.spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype=np.float32
                ),
                observation=gym.spaces.Box(
                    -np.inf, np.inf, shape=obs["observation"].shape, dtype=np.float32
                ),
                robot_observation=gym.spaces.Box(
                    -np.inf, np.inf, shape=obs["robot_observation"].shape, dtype=np.float32
                ),
                image=gym.spaces.Box(-np.inf, np.inf, shape=obs["image"].shape, dtype=np.float32),
            )
        )

    @property
    def task_reward_function(self):
        if getattr(self, "simple_ee_task", False):

            def _reward_fn(achieved, desired, info):
                diff = achieved - desired
                dist = np.linalg.norm(diff, axis=-1)
                reward = -dist
                success = dist < self.success_distance
                reward = np.where(success, self.success_reward, reward)
                if info is not None:
                    try:
                        info.setdefault("is_success_batch", success)
                    except Exception:
                        pass
                return reward

            return _reward_fn
        return self.task.reward_function

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.current_step = 0
        # debug_util.remove_simple_goal_visual(self)
        try:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        except Exception:
            pass
        self.episode_ee_close_steps = 0
        self.world.reset()
        if self.randomization_kwargs["dynamics_randomization"]:
            self.world.apply_domain_randomization(self.randomization_kwargs)

        # Create robot and cloth
        robot_cfg = self.randomization_kwargs["robot"]
        base_pos = robot_cfg["base_pos"]
        base_orn = p.getQuaternionFromEuler(robot_cfg["base_orn_euler"])
        self.robot = PandaRobot(
            base_position=base_pos, base_orientation=base_orn, robot_cfg=robot_cfg
        )
        # Robot dynamics DR
        if self.randomization_kwargs["dynamics_randomization"]:
            _lin = float(np.random.uniform(*robot_cfg["lin_damping_range"]))
            _ang = float(np.random.uniform(*robot_cfg["ang_damping_range"]))
            _frc = float(np.random.uniform(*robot_cfg["lateral_friction_range"]))
            self.robot.randomize_dynamics(_lin, _ang, _frc)
        else:
            _lin = float(robot_cfg["lin_damping"])
            _ang = float(robot_cfg["ang_damping"])
            _frc = float(robot_cfg["lateral_friction"])
            self.robot.randomize_dynamics(_lin, _ang, _frc)

        cloth_cfg = self.randomization_kwargs["cloth"]
        initial_cloth_pos_xy = cloth_cfg["initial_pos"]
        cloth_pos = [
            initial_cloth_pos_xy[0],
            initial_cloth_pos_xy[1],
            self.world.get_table_top_z(),
        ]

        # --- Cloth size ---
        if self.randomization_kwargs["dynamics_randomization"]:
            scale_guess = float(self.np_random.uniform(*cloth_cfg["scale_range"]))
        else:
            scale_guess = float(cloth_cfg["scale"])

        # Clamp to keep Bullet stable
        scale_clip_range = cloth_cfg["scale_clip_range"]
        scale_guess = float(np.clip(scale_guess, scale_clip_range[0], scale_clip_range[1]))

        # Spawn a bit higher for larger cloth to avoid initial interpenetration
        base_clearance = cloth_cfg["base_clearance"]
        extra_clearance_slope = cloth_cfg["extra_clearance_slope"]
        clearance_thresh = cloth_cfg["scale_clearance_threshold"]
        extra_clearance = max(0.0, (scale_guess - clearance_thresh)) * extra_clearance_slope
        cloth_pos[2] = self.world.get_table_top_z() + base_clearance + extra_clearance

        self.cloth = DeformableCloth(
            base_position=cloth_pos,
            cloth_cfg=cloth_cfg,
            randomization_kwargs=self.randomization_kwargs,
            logger=self.logger,
        )

        for _ in range(cloth_cfg["settle_steps"]):
            self.world.step()

        # Update cloth vertices after settling
        self.cloth.update()
        # Reset previous vertices to avoid velocity spike
        self.cloth._prev_verts_raw = self.cloth._cached_verts_raw  # Init history

        # Set camera target
        center_v_name = self.cloth.corner_v_names["mid"]
        center_idx = int(center_v_name.split("_")[1])
        self._camera_target = self.cloth.get_position(center_idx)
        self.camera.begin_episode(self._camera_target)

        if self.simple_ee_task:
            self.task = None
        else:
            self.task = FoldingTask(
                self.task_name,
                self.cloth,
                self.randomization_kwargs,
                self.np_random,
                self.fail_reward,
                self.extra_reward,
                self.goal_noise,
                self.goal_noise_range,
                self.success_distance,
                self.sparse_dense,
                self.success_reward,
            )

        # Move robot to grasp the same corner used as the primary origin site (MuJoCo parity)
        anchor_site_name = None
        if self.task and self.task.constraints:
            anchor_site_name = self.task.constraints[0]["origin"]

        if anchor_site_name and anchor_site_name in self.cloth._site_indices:
            anchor_idx = self.cloth._site_indices[anchor_site_name]
            anchor_v_name = f"v_{anchor_idx}"
        else:
            # Fallback to geometric top-right if something is off
            anchor_v_name = self.cloth.corner_v_names["0"]
            anchor_idx = int(anchor_v_name.split("_")[1])

        anchor_world_pos = self.cloth.get_position(anchor_idx)

        # First pass IK
        joint_positions = self.robot.calculate_ik(anchor_world_pos)
        self.robot.reset_to_joint_positions(joint_positions)

        ee_now = self.robot.get_ee_position_W()
        if np.linalg.norm(anchor_world_pos - ee_now) > 1e-4:
            joint_positions = self.robot.calculate_ik(anchor_world_pos)
            self.robot.reset_to_joint_positions(joint_positions)
            self.world.step()

        self.cloth.create_anchor(anchor_v_name, self.robot.robot_id, self.robot.ee_link_index)

        self.relative_origin = self.robot.get_ee_position_W()
        self.desired_pos_step_W = self.relative_origin.copy()
        self.desired_pos_ctrl_W = self.relative_origin.copy()
        self.min_absolute_W = self.relative_origin + np.array(self.limits_min)
        self.max_absolute_W = self.relative_origin + np.array(self.limits_max)
        self.previous_raw_action = np.zeros_like(self.action_space.sample())
        self.episode_ee_close_steps = 0
        self._prev_ee_pos_W = self.robot.get_ee_position_W()

        if self.simple_ee_task:
            self.goal = self.simple_ee_goal_I.astype(np.float32)
            self.goal_noise = 0.0
        else:
            self.goal, self.goal_noise = self.task.sample_goal(self.get_cloth_position_I())
            # Debug: log origins/targets/goal positions at reset for sanity checks
            if os.getenv("DEBUG_TASK", "0") == "1":
                print("\n[DEBUG_TASK] Constraint info at reset:")
                for i, c in enumerate(self.task.constraints):
                    o = c["origin"]
                    t = c["target"]
                    o_pos = (
                        self.cloth.get_position(self.cloth._site_indices[o]) - self.relative_origin
                    )
                    t_pos = (
                        self.cloth.get_position(self.cloth._site_indices[t]) - self.relative_origin
                    )
                    g_pos = self.goal[i * 3 : (i + 1) * 3]
                    print(
                        f"  {i}: origin={o} pos={o_pos}, target={t} pos={t_pos}, goal={g_pos}, dist={c['distance']}"
                    )

        img = self.get_image_obs()
        self.frame_stack.clear()
        for _ in range(self.frame_stack_size):
            self.frame_stack.append(img)

        if self.randomization_kwargs["materials_randomization"] and not getattr(
            self.cloth, "_tint_applied", False
        ):
            lo = np.array(cloth_cfg["color_lo"])
            hi = np.array(cloth_cfg["color_hi"])
            self.cloth.set_color((self.np_random.uniform(lo, hi)).tolist())

        # ---- Visualize the active task in the GUI (origins, targets, goal rays) ----
        # try:
        #    debug_util.draw_task_visuals(self)
        # except Exception:
        #    pass

        # ---- Spawn a translucent, non-colliding solid box ---
        # debug_util.spawn_workspace_visual_box(
        #     self,
        #     origin=self.relative_origin,
        #     limits_min=self.limits_min,
        #     limits_max=self.limits_max,
        #     rgba=[0.0, 1.0, 0.0, 0.15],
        # )

        # Now show the fully initialized scene (single switch at the very end)
        try:
            if self.has_viewer:
                vcam = self.randomization_kwargs["viewer_debug_camera"]
                p.resetDebugVisualizerCamera(
                    cameraDistance=float(vcam["distance"]),
                    cameraYaw=float(vcam["yaw"]),
                    cameraPitch=float(vcam["pitch"]),
                    cameraTargetPosition=self._camera_target,
                )
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            if self.has_viewer:
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
                p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
                p.configureDebugVisualizer(
                    p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                    int(self.randomization_kwargs["show_depth_preview"]),
                )
                p.configureDebugVisualizer(
                    p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
                    int(self.randomization_kwargs["show_seg_preview"]),
                )
        except Exception:
            pass
        return self.get_obs()

    def step(self, action):
        raw_action = action.copy()
        # Coordinate parity: MuJoCo actions are mirrored relative to Bullet on X.
        # Flip X so MuJoCo demos/policies move the cloth in the same direction here.
        raw_action[0] = -raw_action[0]

        if os.getenv("DEBUG_ACTION", "0") == "1" and self.current_step < 5:
            print(f"[DEBUG_ACTION] step={self.current_step} raw_action(after flip)={raw_action}")
        # 1. Save raw action for observation

        # Capture previous EE position for movement calculation (t-1)
        ee_pos_prev = self._prev_ee_pos_W.copy()

        # 2. Scale action
        action = raw_action * self.output_max

        # 3. Determine random image capture substep
        image_obs_substep_idx = int(
            np.clip(
                np.random.normal(
                    self.image_obs_noise_mean * (self.substeps - 1), self.image_obs_noise_std
                ),
                0,
                self.substeps - 1,
            )
        )

        # 4. Update the "Step" Target (The command step input)
        # This persists across steps!
        self.desired_pos_step_W = np.clip(
            self.desired_pos_step_W + action, self.min_absolute_W, self.max_absolute_W
        )

        if os.getenv("DEBUG_ACTION", "0") == "1" and self.current_step < 5:
            print(
                f"[DEBUG_ACTION] step={self.current_step} desired_pos_step_W={self.desired_pos_step_W} ee_W={self.robot.get_ee_position_W()}"
            )

        # TODO Safety: prevent digging into table (maybe remove)
        table_z = self.world.get_table_top_z()
        if self.desired_pos_step_W[2] < table_z:
            self.desired_pos_step_W[2] = table_z

        # Initialize joint_positions outside loop
        joint_positions = None

        # 5. Substep Loop (Physics & Filter Integration)
        for i in range(self.substeps):
            for _ in range(self.between_steps):
                self.desired_pos_ctrl_W = (
                    self.ctrl_filter * self.desired_pos_step_W
                    + (1 - self.ctrl_filter) * self.desired_pos_ctrl_W
                )

            # OPTIMIZATION: Throttled IK (Every 8 substeps = ~60Hz)
            # if i % 8 == 0:
            #    joint_positions = self.robot.calculate_ik(self.desired_pos_ctrl_W)
            if i == 0:
                joint_positions = self.robot.calculate_ik(self.desired_pos_ctrl_W)

            if joint_positions is not None:
                self.robot.apply_joint_positions(joint_positions)
            self.robot.force_fingers_closed()
            self.world.step()

            if i == image_obs_substep_idx:
                self.frame_stack.append(self.get_image_obs())

        # Update Mesh ONCE per step, just before observations
        self.cloth.update()

        obs = self.get_obs()
        reward, done, info = self._get_reward_and_done(obs, raw_action)

        self.current_step += 1
        self.previous_raw_action = raw_action.copy()

        # OPTIMIZATION: Removed sanity check loop for speed
        # for k in ("image", "observation", "robot_observation", "achieved_goal", "desired_goal"):
        #    if np.any(np.isnan(obs[k])):
        #        raise ValueError(f"NaN in obs['{k}'] detected!")

        return obs, reward, done, info

    def _get_reward_and_done(self, obs, raw_action):
        if self.simple_ee_task:
            # 1. Get positions
            ee_pos_I = self.get_ee_position_I()

            # 2. Distance
            dist_to_target = float(np.linalg.norm(ee_pos_I - self.goal))

            # 3. Success
            is_success = dist_to_target < self.success_distance

            # --- New Pybullet Action Penalty ---
            # Penalize large actions to encourage smoothness/stopping.
            action_penalty = np.sum(np.square(raw_action)) * 0.1

            # 4. Dense Reward
            if is_success:
                # We still apply penalty during success so it learns to hold still!
                reward = (self.success_reward + 1.0) - action_penalty
                self.episode_ee_close_steps += 1
            else:
                reward = -dist_to_target - action_penalty
                self.episode_ee_close_steps = 0

            # 5. Done logic
            done = self.episode_ee_close_steps >= self.max_close_steps

            info = {
                "reward": float(reward),
                "is_success": bool(is_success),
                "dist_to_target": dist_to_target,
                "action_penalty": float(action_penalty),
                "delta_size": float(np.linalg.norm(raw_action)),
                "ctrl_error": float(
                    np.linalg.norm(self.desired_pos_ctrl_W - self.robot.get_ee_position_W())
                ),
                "corner_sum_error": dist_to_target,
                "corner_positions": np.zeros(8, dtype=np.float32),
                "corner_0": dist_to_target,
                "corner_1": 0.0,
                "corner_2": 0.0,
                "corner_3": 0.0,
                "corner_distance": dist_to_target,
                "env_memory_usage": self.process.memory_info().rss if self.process else 0,
                "ee_target_W": self.desired_pos_ctrl_W.copy(),
                "ee_target_step_W": self.desired_pos_step_W.copy(),
                "ee_W": self.robot.get_ee_position_W().copy(),
            }
            return reward, done, info

        # 1. Action Penalty
        action_penalty = np.sum(np.square(raw_action)) * 0.1

        # 2. Task Reward
        task_reward = self.task.compute_reward(obs["achieved_goal"], self.goal, {})
        reward = task_reward - action_penalty

        # --- OPTIMIZATION: Use direct integer access (FAST) ---
        # Distances per constraint origin → its goal vector (MuJoCo parity)
        constraint_dists = {}
        for i, c in enumerate(self.task.constraints):
            site_name = c["origin"]
            v_id = self.cloth._site_indices.get(site_name, None)
            if v_id is None:
                constraint_dists[site_name] = 0.0
                continue
            achieved_pos_I = self.cloth.get_position(v_id) - self.relative_origin
            target_pos_I = self.goal[i * 3 : (i + 1) * 3]
            constraint_dists[site_name] = float(np.linalg.norm(achieved_pos_I - target_pos_I))

        if os.getenv("DEBUG_TASK_STEP", "0") == "1" and self.current_step < 5:
            print(f"[DEBUG_TASK_STEP] step={self.current_step}")
            for i, c in enumerate(self.task.constraints):
                site_name = c["origin"]
                v_id = self.cloth._site_indices.get(site_name, None)
                achieved_pos_I = (
                    self.cloth.get_position(v_id) - self.relative_origin
                    if v_id is not None
                    else None
                )
                target_pos_I = self.goal[i * 3 : (i + 1) * 3]
                print(
                    f"  {site_name}: achieved={achieved_pos_I}, target={target_pos_I}, dist={constraint_dists.get(site_name)}"
                )

        # Map the primary corner metrics to the first two corner constraints (sideways uses S8_8, S0_8)
        distances = {
            "0": constraint_dists.get("S0_8", 0.0),
            "1": constraint_dists.get("S8_8", 0.0),
            "2": constraint_dists.get("S0_0", 0.0),
            "3": constraint_dists.get("S8_0", 0.0),
        }
        dist_to_target = distances.get("1", 0.0)
        is_success = task_reward > self.fail_reward

        corner_pixels = self._get_corner_image_positions().astype(np.float32)

        info = {
            "reward": float(reward),
            "is_success": bool(is_success),
            "delta_size": float(np.linalg.norm(raw_action)),
            "action_penalty": float(action_penalty),
            "ctrl_error": float(
                np.linalg.norm(self.desired_pos_ctrl_W - self.robot.get_ee_position_W())
            ),
            "corner_sum_error": 0.0,
            "corner_positions": corner_pixels.reshape(-1).astype(np.float32),
            "env_memory_usage": self.process.memory_info().rss if self.process else 0,
            "ee_target_W": self.desired_pos_ctrl_W.copy(),
            "ee_target_step_W": self.desired_pos_step_W.copy(),
            "ee_W": self.robot.get_ee_position_W().copy(),
        }

        for k in distances:
            info[f"corner_{k}"] = distances[k]
            info["corner_sum_error"] += distances[k]

        info["corner_distance"] = dist_to_target
        info["dsum"] = info["corner_sum_error"]

        if dist_to_target < self.success_distance:
            self.episode_ee_close_steps += 1
        else:
            self.episode_ee_close_steps = 0

        done = self.episode_ee_close_steps >= self.max_close_steps
        info["is_success"] = bool(is_success)

        return reward, done, info

    def _project_points_uv(self, points_W, label="points", cam_type="default"):
        if cam_type == "eval_camera":
            cam_setup = self.camera.get_stable_camera_setup(self._camera_target, cam_type=cam_type)
            view, proj = self.camera.get_stable_view_projection_matrices(
                self._camera_target, cam_type=cam_type
            )
        else:
            cam_setup = {"type": "episode"}
            view, proj = self.camera.get_view_projection_matrices(self._camera_target)

        V = np.array(view, dtype=np.float64).reshape(4, 4).T
        P = np.array(proj, dtype=np.float64).reshape(4, 4).T
        VP = P @ V

        W_render, H_render = self.camera.render_size

        per_point = []
        uv_full = []
        for x, y, z in points_W:
            pos_h = np.array([x, y, z, 1.0], dtype=np.float64)
            clip = VP @ pos_h
            if abs(clip[3]) < 1e-12:
                per_point.append({"world": [float(x), float(y), float(z)], "invalid": True})
                uv_full.extend([0.0, 0.0])
                continue
            ndc = (clip[:3] / clip[3]).astype(np.float64)
            x_pix = (ndc[0] + 1.0) * 0.5 * W_render
            y_pix = (1.0 - ndc[1]) * 0.5 * H_render
            u_full = float(np.clip(x_pix / max(1, W_render), 0.0, 1.0))
            v_full = float(np.clip(y_pix / max(1, H_render), 0.0, 1.0))
            per_point.append(
                {
                    "world": [float(x), float(y), float(z)],
                    "clip": [float(c) for c in clip.tolist()],
                    "ndc": [float(n) for n in ndc.tolist()],
                    "pix_full": [float(x_pix), float(y_pix)],
                    "uv_full": [u_full, v_full],
                }
            )
            uv_full.extend([u_full, v_full])

        out = {
            "label": label,
            "camera": cam_setup,
            "uv_full": uv_full,
            "points": per_point,
        }
        return out

    def _get_corner_image_positions(self, cam_type="default"):
        # Same order as MuJoCo corner_index_mapping:
        # 0: S0_8, 1: S8_8, 2: S0_0, 3: S8_0
        site_order = ["S0_8", "S8_8", "S0_0", "S8_0"]

        pts = []
        for s in site_order:
            idx = self.cloth._site_indices.get(s)
            if idx is None:
                # Fallback: if for some reason the site is missing, fall back to geometric corner
                # (shouldn't normally happen)
                return np.array([], dtype=np.float32)
            pts.append(self.cloth.get_position(idx))

        log = self._project_points_uv(pts, label="corners", cam_type=cam_type)
        return np.array(log["uv_full"], dtype=np.float32)

    def get_image_obs(self):
        return self.camera.policy_image(self._camera_target)

    def get_obs(self):
        ee_pos_W = self.robot.get_ee_position_W()
        ee_vel_W = self.get_ee_velocity()

        # Calculate standard EE info
        self._prev_ee_pos_W = ee_pos_W
        ee_pos_I = ee_pos_W - self.relative_origin
        desired_pos_ctrl_I = self.desired_pos_ctrl_W - self.relative_origin

        if self.simple_ee_task:
            # 1. Calculate dimensions exactly like the full task to match network size
            n_sites = len(self.cloth.sites)
            cloth_dim = n_sites * 3 * 2
            physics_dim = 9 if self.randomization_kwargs["dynamics_randomization"] else 0
            cloth_obs = np.zeros(cloth_dim + physics_dim, dtype=np.float32)
            cloth_obs[0:3] = ee_pos_I
            cloth_obs[3:6] = ee_vel_W
            achieved_goal = ee_pos_I.astype(np.float32)
        else:
            cloth_pos_I, vel_sites_W = self.cloth.get_site_observations(
                self.timestep, self.relative_origin
            )
            achieved_goal = self.task.get_achieved_goal(cloth_pos_I)
            cloth_obs = np.concatenate(
                [
                    np.array(list(cloth_pos_I.values())).flatten(),
                    np.array(list(vel_sites_W.values())).flatten(),
                ]
            )
            if self.randomization_kwargs["dynamics_randomization"]:
                g = float(self.world.gravity)
                tab_mu = float(self.world.table_lateral_friction)
                tab_e = float(self.world.table_restitution)
                jd = float(np.mean(self.robot.joint_damping))
                jf = float(np.mean(self.robot.joint_friction))

                physics_params = [
                    g,
                    tab_mu,
                    tab_e,
                    float(self.cloth.frictionCoeff),
                    float(getattr(self.cloth, "thickness", 0.002)),
                    float(getattr(self.cloth, "springElasticStiffness", 40.0)),
                    float(getattr(self.cloth, "springDampingStiffness", 0.1)),
                    jd,
                    jf,
                ]
                cloth_obs = np.concatenate([cloth_obs, np.array(physics_params, dtype=np.float32)])

        if self.robot_observation == "ee":
            robot_obs = np.concatenate([ee_pos_I, ee_vel_W, desired_pos_ctrl_I])
        elif self.robot_observation == "ctrl":
            robot_obs = np.concatenate([self.previous_raw_action, np.zeros(6)])
        elif self.robot_observation == "none":
            robot_obs = np.zeros(9, dtype=np.float32)

        image_stack = np.array(list(self.frame_stack)).flatten()

        def nan(a):
            return np.nan_to_num(a, nan=0.0, posinf=1e3, neginf=-1e3)

        def clip(a, mi, ma):
            return np.clip(a, mi, ma)

        return {
            "achieved_goal": clip(nan(achieved_goal), -1e3, 1e3).copy(),
            "desired_goal": clip(nan(self.goal.astype(np.float32)), -1e3, 1e3).copy(),
            "image": clip(
                np.nan_to_num(image_stack, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0
            ).copy(),
            "observation": clip(nan(cloth_obs), -1e3, 1e3).copy().flatten(),
            "robot_observation": clip(nan(robot_obs), -1e3, 1e3).copy().flatten(),
        }

    def get_ee_position_W(self):
        return self.robot.get_ee_position_W()

    def get_ee_position_I(self):
        return self.robot.get_ee_position_W() - self.relative_origin

    def get_ee_velocity(self):
        ls = p.getLinkState(self.robot.robot_id, self.robot.ee_link_index, computeLinkVelocity=1)
        return np.array(ls[6], dtype=np.float32)

    def get_joint_positions(self):
        return self.robot.get_joint_positions()

    def get_joint_velocities(self):
        return self.robot.get_joint_velocities()

    def get_cloth_position_I(self):
        positions = {}
        for site, idx in self.cloth._site_indices.items():
            positions[site] = self.cloth.get_position(idx) - self.relative_origin
        return positions

    def get_masked_image(
        self,
        point_size=5,
        greyscale=False,
        aux_output=None,
        img=None,
        camera_type="default",
        ee_in_image=None,
    ):
        if img is None:
            img = self.camera.render_rgb(self._camera_target, cam_type=camera_type)
        h, w = img.shape[:2]

        # (MuJoCo parity) Project EE into image if provided.
        if ee_in_image is not None:
            ee_w = np.asarray(ee_in_image, dtype=np.float64)[:3].reshape(1, 3)
            ee_log = self._project_points_uv(ee_w, label="ee", cam_type=camera_type)
            u_ee, v_ee = ee_log["uv_full"][:2]
            cv2.circle(img, (int(u_ee * w), int(v_ee * h)), int(point_size + 2), (0, 0, 0), -1)

        # Blue: GT corners in [0,1]
        uv = self._get_corner_image_positions(cam_type=camera_type)
        for i in range(0, len(uv), 2):
            u = int(np.clip(uv[i] * w, 0, w - 1))
            v = int(np.clip(uv[i + 1] * h, 0, h - 1))
            cv2.circle(img, (u, v), int(point_size), (255, 0, 0), -1)

        # Green: predicted (assumed already in [0,1]); use first 8 values (4 uv pairs)
        if aux_output is not None:
            flat = np.asarray(aux_output, dtype=np.float32).flatten()[:8]
            flat = np.clip(flat, 0.0, 1.0)
            for i in range(0, min(len(flat), 8), 2):
                au = int(np.clip(flat[i] * w, 0, w - 1))
                av = int(np.clip(flat[i + 1] * h, 0, h - 1))
                cv2.circle(img, (au, av), int(point_size), (0, 255, 0), -1)

        if greyscale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img

    def capture_images(self, aux_output=None):
        if aux_output is None:
            print("[Bullet] capture_images: aux_output is None", flush=True)

        ee_in_image = np.ones(4)
        ee_pos = self.get_ee_position_W()
        ee_in_image[:3] = ee_pos

        policy_rgb_big = self.camera.render_rgb_dr(
            self._camera_target, size=self.camera.render_size
        )
        policy_rgb_small = self.camera.render_rgb_dr(self._camera_target)

        corner_image = self.get_masked_image(
            point_size=8,
            greyscale=False,
            aux_output=aux_output,
            img=policy_rgb_big,
            camera_type="default",
            ee_in_image=ee_in_image,
        )

        cnn_color_image_full = self.get_masked_image(
            point_size=2,
            greyscale=False,
            aux_output=aux_output,
            img=policy_rgb_big,
            camera_type="default",
            ee_in_image=ee_in_image,
        )

        cnn_color_image = self.get_masked_image(
            point_size=2,
            greyscale=False,
            aux_output=aux_output,
            img=policy_rgb_small,
            camera_type="default",
            ee_in_image=ee_in_image,
        )

        cnn_image = self.get_masked_image(
            point_size=2,
            greyscale=True,
            aux_output=aux_output,
            img=policy_rgb_small,
            camera_type="default",
            ee_in_image=ee_in_image,
        )

        eval_image_rgb = self.camera.render_rgb_full(self._camera_target, cam_type="eval_camera")
        eval_image = self.get_masked_image(
            point_size=4,
            greyscale=False,
            aux_output=None,
            img=eval_image_rgb,
            camera_type="eval_camera",
            ee_in_image=ee_in_image,
        )

        return (corner_image, eval_image, cnn_color_image_full, cnn_color_image, cnn_image)

    def get_trajectory_log_entry(self):
        return {
            "origin": self.relative_origin,
            "output_max": self.output_max,
            "desired_pos_step_I": self.desired_pos_step_W - self.relative_origin,
            "desired_pos_ctrl_I": self.desired_pos_ctrl_W - self.relative_origin,
            "ee_position_I": self.robot.get_ee_position_W() - self.relative_origin,
            "raw_action": self.previous_raw_action,
            "substeps": self.substeps,
            "timestep": self.timestep,
            "goal_noise": self.goal_noise,
        }

    def close(self):
        self.world.close()


class ClothEnvBullet(BulletClothEnv_, EzPickle):
    """Public class to mirror ClothEnv signature and be EzPickle-compatible."""

    def __init__(self, **kwargs):
        BulletClothEnv_.__init__(self, **kwargs)
        EzPickle.__init__(self, **kwargs)
