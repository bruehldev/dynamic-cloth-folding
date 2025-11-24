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
        # Allows bypassing the cloth FoldingTask with a static EE target in I-coordinates.
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
        """
        Provides the task_reward_function for compatibility with the training script,
        forwarding it from the internal FoldingTask instance. When a simple
        EE-reaching task is active, return a small closure implementing that reward.
        """
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
        # self._remove_simple_goal_visual()
        # Hide intermediate loads & speed up reset (single guard)
        try:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        except Exception:
            pass
        self.episode_ee_close_steps = 0
        self.world.reset()
        if self.randomization_kwargs[
            "dynamics_randomization"
        ]:  # Check if dynamics randomization is good name
            self.world.apply_domain_randomization(self.randomization_kwargs)

        # Create robot and cloth
        robot_cfg = self.randomization_kwargs["robot"]
        base_pos = robot_cfg["base_pos"]
        base_orn = p.getQuaternionFromEuler(robot_cfg["base_orn_euler"])
        self.robot = PandaRobot(
            base_position=base_pos, base_orientation=base_orn, robot_cfg=robot_cfg
        )
        # Robot dynamics DR (only if master switch is ON)
        if self.randomization_kwargs["dynamics_randomization"]:
            _lin = float(np.random.uniform(*robot_cfg["lin_damping_range"]))
            _ang = float(np.random.uniform(*robot_cfg["ang_damping_range"]))
            _frc = float(np.random.uniform(*robot_cfg["lateral_friction_range"]))
            # Prefer the PandaRobot helper if present; call positionally for max compatibility.
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

        # ---- Cloth domain randomization (physics + size + optional color) ----

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

        # Update cloth vertices after settling so IK and obs are correct
        self.cloth.update()
        # Reset previous vertices to current to avoid massive velocity spike on first step
        self.cloth._prev_verts_W = self.cloth.get_raw_vertex_positions().copy()

        # Set camera target to the cloth's center, matching MuJoCo's lookatbody
        center_v_name = self.cloth.corner_v_names["mid"]
        center_idx = int(center_v_name.split("_")[1])
        self._camera_target = self.cloth.get_position(center_idx)
        self.camera.begin_episode(self._camera_target)

        # Initialize the task (optional simple EE goal bypasses FoldingTask)
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

        # Move robot to grasp corner
        corner_v_name = self.cloth.corner_v_names["0"]
        corner_idx = int(corner_v_name.split("_")[1])
        corner_world_pos = self.cloth.get_position(corner_idx)

        # First pass IK
        joint_positions = self.robot.calculate_ik(corner_world_pos)
        self.robot.reset_to_joint_positions(joint_positions)

        # Check error and do a second pass only if needed.
        ee_now = self.robot.get_ee_position_W()
        if np.linalg.norm(corner_world_pos - ee_now) > 1e-4:
            joint_positions = self.robot.calculate_ik(corner_world_pos)
            self.robot.reset_to_joint_positions(joint_positions)
            self.world.step()

        # Ensure orientation is locked before making the soft anchor (silent unless summary below)
        """
        ee_quat = self.robot.get_ee_pose_W()
        orn_err_deg = self.robot.quat_angle_error_deg(ee_quat, self.robot._ik_target_quat)
        if orn_err_deg > 0.5:  # ~0.5° tolerance
            joint_positions = self.robot.calculate_ik(corner_world_pos)
            self.robot.reset_to_joint_positions(joint_positions)
            self.world.step()
            ee_quat = self.robot.get_ee_pose_W()
            orn_err_deg = self.robot.quat_angle_error_deg(ee_quat, self.robot._ik_target_quat)
        """
        # Anchor cloth to robot's hand (using the correct ee_link_index)
        self.cloth.create_anchor(corner_v_name, self.robot.robot_id, self.robot.ee_link_index)

        # Initialize state variables based on final EE position
        self.relative_origin = self.robot.get_ee_position_W()
        self.desired_pos_step_W = self.relative_origin.copy()
        self.desired_pos_ctrl_W = self.relative_origin.copy()
        self.min_absolute_W = self.relative_origin + np.array(self.limits_min)
        self.max_absolute_W = self.relative_origin + np.array(self.limits_max)
        self.previous_raw_action = np.zeros_like(self.action_space.sample())
        self.episode_ee_close_steps = 0
        self._prev_ee_pos_W = self.robot.get_ee_position_W()

        # Set goal for the episode
        if self.simple_ee_task:
            self.goal = self.simple_ee_goal_I.astype(np.float32)
            self.goal_noise = 0.0
        else:
            self.goal, self.goal_noise = self.task.sample_goal(self.get_cloth_position_I())

        # self._update_simple_goal_visual()

        # Capture initial image (viewer can stay off; camera grabs directly)
        img = self.get_image_obs()
        self.frame_stack.clear()
        for _ in range(self.frame_stack_size):
            self.frame_stack.append(img)

        # Randomize cloth color (skip if DeformableCloth already tinted)
        if self.randomization_kwargs["materials_randomization"] and not getattr(
            self.cloth, "_tint_applied", False
        ):
            lo = np.array(cloth_cfg["color_lo"])
            hi = np.array(cloth_cfg["color_hi"])
            self.cloth.set_color((self.np_random.uniform(lo, hi)).tolist())

        # ---- Visualize the active task in the GUI (origins, targets, goal rays) ----
        # try:
        #    self._draw_task_visuals()
        # except Exception:
        #    pass

        # ---- Spawn a translucent, non-colliding solid box ---
        # self._spawn_workspace_visual_box(
        #    origin=self.relative_origin,
        #    limits_min=self.limits_min,
        #    limits_max=self.limits_max,
        #    rgba=[0.0, 1.0, 0.0, 0.15],
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
            # Turn rendering back on…
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            # …and re-enable preview panes (RGB on by default; depth/seg off unless requested)
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

        # TODO Safety: prevent digging into table (maybe remove)
        table_z = self.world.get_table_top_z()
        if self.desired_pos_step_W[2] < table_z:
            self.desired_pos_step_W[2] = table_z

        # Initialize joint_positions outside loop
        joint_positions = None

        # 5. Substep Loop (Physics & Filter Integration)
        for i in range(self.substeps):
            # Simulate the analog low-pass filter (capacitor)
            # We iterate `between_steps` times to smooth the signal
            for _ in range(self.between_steps):
                self.desired_pos_ctrl_W = (
                    self.ctrl_filter * self.desired_pos_step_W
                    + (1 - self.ctrl_filter) * self.desired_pos_ctrl_W
                )

            # TODO OPTIMIZATION: Throttled IK (Every 4 substeps = ~120Hz)
            # This drastically reduces CPU load. (maybe remove)
            if i % 4 == 0:
                joint_positions = self.robot.calculate_ik(self.desired_pos_ctrl_W)

            # 7. Apply control and step physics
            if joint_positions is not None:
                self.robot.apply_joint_positions(joint_positions)
            self.robot.force_fingers_closed()
            self.world.step()

            self.cloth.update()

            # 8. Capture image if it's the right substep
            if i == image_obs_substep_idx:
                self.frame_stack.append(self.get_image_obs())

        obs = self.get_obs()
        reward, done, info = self._get_reward_and_done(obs, raw_action)

        self.current_step += 1
        self.previous_raw_action = raw_action.copy()

        # try:
        #    if self.has_viewer:
        #        self.camera.print_gui_camera_as_type(name=f"step_{self.current_step:04d}")
        # except Exception:
        #    pass

        # Sanity Check for NaNs
        for k in ("image", "observation", "robot_observation", "achieved_goal", "desired_goal"):
            if np.any(np.isnan(obs[k])):
                raise ValueError(f"NaN in obs['{k}'] detected!")

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
            # 0.1 is a common weight. If raw_action is [1,1,1], penalty is 0.3.
            # If raw_action is [0,0,0], penalty is 0.
            action_penalty = np.sum(np.square(raw_action)) * 0.1
            # --- FIX END ---

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
                "action_penalty": float(action_penalty),  # Good for debugging logs
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
        # REMOVED: verts_W = self.cloth.get_positions_W()

        distances = {}
        all_targets_I = [self.goal[i * 3 : (i + 1) * 3] for i in range(len(self.task.constraints))]

        site_name_to_goal_idx = {c["origin"]: i for i, c in enumerate(self.task.constraints)}

        for corner_key in ("0", "1", "2", "3"):
            # Get the Vertex ID directly from string "v_123" -> 123
            v_str = self.cloth.corner_v_names[corner_key]
            v_id = int(v_str.split("_")[1])

            # FAST ACCESS: O(1)
            achieved_pos_I = self.cloth.get_position(v_id) - self.relative_origin

            # Find site name for this vertex index
            site_name = None
            for s_name, s_idx in self.cloth._site_indices.items():
                if s_idx == v_id:
                    site_name = s_name
                    break

            if site_name and site_name in site_name_to_goal_idx:
                goal_idx = site_name_to_goal_idx[site_name]
                target_pos_I = self.goal[goal_idx * 3 : (goal_idx + 1) * 3]
                distances[corner_key] = float(np.linalg.norm(achieved_pos_I - target_pos_I))
            elif all_targets_I:
                distances[corner_key] = float(
                    min(np.linalg.norm(achieved_pos_I - t) for t in all_targets_I)
                )
            else:
                distances[corner_key] = 0.0

        dist_to_target = distances["1"]
        is_success = task_reward > self.fail_reward

        # Get corner positions for visualization (fast access)
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
        """
        Project world points to normalized UVs [0,1] using the SAME camera as the image.
        For 'default' views, use the episode/policy camera. For 'eval_camera', use the stable view.
        """
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

        # Normalize by the target image buffer we drew into
        W_render, H_render = self.camera.render_size  # 500x500 for big, 100x100 for small

        per_point = []
        uv_full = []
        for x, y, z in points_W:
            pos_h = np.array([x, y, z, 1.0], dtype=np.float64)
            clip = VP @ pos_h
            if abs(clip[3]) < 1e-12:
                per_point.append({"world": [float(x), float(y), float(z)], "invalid": True})
                uv_full.extend([0.0, 0.0])
                continue
            ndc = (clip[:3] / clip[3]).astype(np.float64)  # (-1..1)
            x_pix = (ndc[0] + 1.0) * 0.5 * W_render
            y_pix = (1.0 - ndc[1]) * 0.5 * H_render  # flip Y
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

    # ---------- CORNER UVs ----------
    def _get_corner_image_positions(self, cam_type="default"):
        names = ["0", "1", "2", "3"]
        named_pts = []
        for n in names:
            v_str = self.cloth.corner_v_names[n]  # e.g. "v_123"
            v_id = int(v_str.split("_")[1])
            named_pts.append(self.cloth.get_position(v_id))

        named_log = self._project_points_uv(named_pts, label="named_corners", cam_type=cam_type)
        return np.array(named_log["uv_full"], dtype=np.float32)

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
            cloth_dim = n_sites * 3 * 2  # pos + vel
            physics_dim = 9 if self.randomization_kwargs["dynamics_randomization"] else 0

            # 2. Create the zero vector
            cloth_obs = np.zeros(cloth_dim + physics_dim, dtype=np.float32)

            # 3. Inject EE state
            cloth_obs[0:3] = ee_pos_I  # Position
            cloth_obs[3:6] = ee_vel_W  # Velocity

            achieved_goal = ee_pos_I.astype(np.float32)
        else:
            # --- OPTIMIZATION: Use fast path ---
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

            # --- Physics DR scalars (MuJoCo parity) ---
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
        # Retrieve linear velocity (index 6) from LinkState
        # computeLinkVelocity=1 is required in recent PyBullet versions
        ls = p.getLinkState(self.robot.robot_id, self.robot.ee_link_index, computeLinkVelocity=1)
        return np.array(ls[6], dtype=np.float32)

    def get_joint_positions(self):
        return self.robot.get_joint_positions()

    def get_joint_velocities(self):
        return self.robot.get_joint_velocities()

    def get_cloth_position_I(self):
        # Optimized to use _site_indices directly
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
        """
        Draw GT (blue) + optional aux (green) corner dots on an image (no crop).
        If 'img' is None, a stable RGB frame is rendered at the camera's full size.
        """
        # Stable RGB unless a specific image is provided
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
            # If predictions look like pixels (e.g., ~[0..100]), normalize to [0,1]
            # if np.nanmax(flat) > 1.0:
            #    W_ref, H_ref = self.image_size  # policy/CNN input size
            #    flat[0::2] = flat[0::2] / float(W_ref)
            #    flat[1::2] = flat[1::2] / float(H_ref)
            flat = np.clip(flat, 0.0, 1.0)
            for i in range(0, min(len(flat), 8), 2):
                au = int(np.clip(flat[i] * w, 0, w - 1))
                av = int(np.clip(flat[i + 1] * h, 0, h - 1))
                cv2.circle(img, (au, av), int(point_size), (0, 255, 0), -1)

        if greyscale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img

    def capture_images(self, aux_output=None):
        """
        Mirrors MuJoCo's capture_images():
        returns (corner_image, eval_image, cnn_color_image_full, cnn_color_image, cnn_image).
        """
        if aux_output is None:
            print("[Bullet] capture_images: aux_output is None", flush=True)

        ee_in_image = np.ones(4)
        ee_pos = self.get_ee_position_W()
        ee_in_image[:3] = ee_pos

        # Render the SAME policy/off-screen camera at both sizes (no crop)
        policy_rgb_big = self.camera.render_rgb_dr(
            self._camera_target, size=self.camera.render_size
        )
        policy_rgb_small = self.camera.render_rgb_dr(self._camera_target)  # defaults to image_size

        # 0) Corner overlay (DEFAULT view), larger dots for visibility (8px)
        corner_image = self.get_masked_image(
            point_size=8,
            greyscale=False,
            aux_output=aux_output,
            img=policy_rgb_big,
            camera_type="default",
            ee_in_image=ee_in_image,
        )

        # 2) CNN color image (FULL render buffer), tiny dots (2px)
        cnn_color_image_full = self.get_masked_image(
            point_size=2,
            greyscale=False,
            aux_output=aux_output,
            img=policy_rgb_big,
            camera_type="default",
            ee_in_image=ee_in_image,
        )

        # 3) CNN color image (policy size, no crop)
        cnn_color_image = self.get_masked_image(
            point_size=2,
            greyscale=False,
            aux_output=aux_output,
            img=policy_rgb_small,
            camera_type="default",
            ee_in_image=ee_in_image,
        )

        # 4) CNN grayscale image (policy size, no crop)
        cnn_image = self.get_masked_image(
            point_size=2,
            greyscale=True,
            aux_output=aux_output,
            img=policy_rgb_small,
            camera_type="default",
            ee_in_image=ee_in_image,
        )
        # Use the policy_image pipeline to get the exact grayscale
        # cnn_image_flat = self.camera.policy_image(self._camera_target)  # [0,1], flat
        # cnn_image = (cnn_image_flat.reshape(h_cnn, w_cnn) * 255.0).astype("uint8")

        # 1) Eval image from eval_camera (stable), smaller dots (4px), no aux overlay)
        # Do this LAST so the GUI preview (if enabled) settles on the eval view.
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
        """Returns a dictionary of info for logging, matching original keys."""
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

    def _remove_simple_goal_visual(self):
        if not getattr(self, "_simple_goal_debug_ids", None):
            return
        for _id in self._simple_goal_debug_ids:
            with np.errstate(all="ignore"):
                p.removeUserDebugItem(_id)
        self._simple_goal_debug_ids = []

    def _update_simple_goal_visual(self):
        self._remove_simple_goal_visual()
        if not getattr(self, "simple_ee_task", False):
            return
        goal_I = np.asarray(self.goal, dtype=np.float32)
        goal_W = (self.relative_origin + goal_I).astype(np.float32)
        goal_tip = (goal_W + np.array([0.0, 0.0, 0.08], dtype=np.float32)).tolist()
        color = [1.0, 0.3, 0.1]
        self._simple_goal_debug_ids = []
        self._simple_goal_debug_ids.append(
            p.addUserDebugPoints([goal_W.tolist()], [color], pointSize=12, lifeTime=0)
        )
        self._simple_goal_debug_ids.append(
            p.addUserDebugLine(goal_W.tolist(), goal_tip, color, lineWidth=3.0, lifeTime=0)
        )
        self._simple_goal_debug_ids.append(
            p.addUserDebugText(
                "EE goal",
                goal_tip,
                textColorRGB=color,
                textSize=1.4,
                lifeTime=0,
            )
        )

    def _spawn_workspace_visual_box(self, origin, limits_min, limits_max, rgba=[0, 1, 0, 0.15]):
        """Create a translucent box that matches the workspace. No collisions."""
        # Remove previous visual box if it exists
        if getattr(self, "_ws_vis_id", None) is not None:
            try:
                p.removeBody(self._ws_vis_id)
            except Exception:
                pass
            self._ws_vis_id = None

        o = np.array(origin, dtype=float)
        mn = np.array(limits_min, dtype=float)
        mx = np.array(limits_max, dtype=float)
        # Be robust to swapped min/max:
        lo = np.minimum(mn, mx)
        hi = np.maximum(mn, mx)
        half_extents = (hi - lo) * 0.5
        center = o + (lo + hi) * 0.5

        vis = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=half_extents.tolist(),
            rgbaColor=rgba,
        )
        self._ws_vis_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vis,
            basePosition=center.tolist(),
            baseOrientation=[0, 0, 0, 1],
        )

    # ---------------- Task visualization helpers (non-physics) ----------------
    def _draw_task_visuals(self):
        """Draws: (1) origin→target line per constraint, (2) origin→goal ray, (3) small spheres on sites."""
        if self.task is None:
            return
        # Clear previous
        for _id in getattr(self, "_task_line_ids", []):
            with np.errstate(all="ignore"):
                p.removeUserDebugItem(_id)
        for bid in getattr(self, "_task_marker_ids", []):
            with np.errstate(all="ignore"):
                p.removeBody(bid)
        self._task_line_ids, self._task_marker_ids = [], []

        # Colors
        col_origin = [0.1, 0.6, 1.0]  # blue-ish
        col_target = [1.0, 0.8, 0.1]  # yellow-ish
        col_goal_ray = [0.2, 1.0, 0.4]  # green-ish
        line_w = 2.0

        # Tiny sphere visual shape (reused)
        sph_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.008, rgbaColor=[1, 1, 1, 1])

        for ci, c in enumerate(self.task.constraints):
            ok = c["origin"]
            tk = c["target"]  # site names (e.g., "S0_8")

            idx_o = self.cloth._site_indices[ok]
            idx_t = self.cloth._site_indices[tk]
            o = np.array(self.cloth.get_position(idx_o), dtype=float)
            t = np.array(self.cloth.get_position(idx_t), dtype=float)

            # 1) Origin → Target line (white-ish to distinguish)
            self._task_line_ids.append(
                p.addUserDebugLine(
                    o.tolist(), t.tolist(), [0.9, 0.9, 0.9], lineWidth=line_w, lifeTime=0
                )
            )

            # 2) Origin → Goal ray (distance per constraint, no noise)
            dvec = t - o  # direction to target
            nrm = np.linalg.norm(dvec)
            if nrm > 1e-9:
                dvec = dvec / nrm  # unit
            g = o + dvec * float(c["distance"])
            self._task_line_ids.append(
                p.addUserDebugLine(
                    o.tolist(), g.tolist(), col_goal_ray, lineWidth=line_w, lifeTime=0
                )
            )

            # 3) Small non-colliding markers on origin (blue) and target (yellow)
            self._task_marker_ids.append(
                p.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=sph_vis,
                    basePosition=o.tolist(),
                )
            )
            p.changeVisualShape(self._task_marker_ids[-1], -1, rgbaColor=col_origin + [1.0])
            self._task_marker_ids.append(
                p.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=sph_vis,
                    basePosition=t.tolist(),
                )
            )
            p.changeVisualShape(self._task_marker_ids[-1], -1, rgbaColor=col_target + [1.0])

            # 4) Labels (origin/target indices)
            self._task_line_ids.append(
                p.addUserDebugText(
                    f"{ok}", o.tolist(), textColorRGB=col_origin, textSize=1.2, lifeTime=0
                )
            )
            self._task_line_ids.append(
                p.addUserDebugText(
                    f"{tk}", t.tolist(), textColorRGB=col_target, textSize=1.2, lifeTime=0
                )
            )

        # Keep the tiny sphere visual handle alive (it’s owned by markers; OK to leave)


class ClothEnvBullet(BulletClothEnv_, EzPickle):
    """Public class to mirror ClothEnv signature and be EzPickle-compatible."""

    def __init__(self, **kwargs):
        BulletClothEnv_.__init__(self, **kwargs)
        EzPickle.__init__(self, **kwargs)
