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
        save_folder=None,
        has_viewer=False,
        logger: Optional[Any] = None,
        **kwargs,
    ):
        self.logger = logger if logger is not None else _NoOpLogger()

        if current_process().name != "MainProcess":
            has_viewer = False
            self.logger.log("Viewer disabled in subprocess.")

        self.process = psutil.Process(os.getpid())
        self.seed()

        # --- Init Params from kwargs ---
        self.kwargs = randomization_kwargs
        self.kwargs.update(kwargs)  # Merge env_kwargs

        task_cfg = self.kwargs["folding_task"]
        self.task_name = self.kwargs["task_name"]
        self.save_folder = save_folder
        self.timestep = float(self.kwargs["timestep"])
        self.control_frequency = float(self.kwargs["control_frequency"])
        self.substeps = max(1, int(1.0 / (self.timestep * self.control_frequency)))
        self.filter = float(self.kwargs["ctrl_filter"])
        steps_per_second = 1.0 / self.timestep
        self.between_steps = int(1000.0 / steps_per_second)
        self.output_max = float(self.kwargs["output_max"])
        self.robot_observation = str(self.kwargs["robot_observation"])
        self.max_close_steps = int(self.kwargs["max_close_steps"])
        self.success_distance = float(task_cfg["success_distance"])
        self.frame_stack_size = int(self.kwargs["frame_stack_size"])
        self.sparse_dense = task_cfg["sparse_dense"]
        self.success_reward = task_cfg["success_reward"]
        self.fail_reward = task_cfg["fail_reward"]
        self.extra_reward = task_cfg["extra_reward"]
        self.image_size = (self.kwargs["image_size"], self.kwargs["image_size"])

        self.has_viewer = has_viewer
        self.image_obs_noise_mean = self.kwargs["image_obs_noise_mean"]
        self.image_obs_noise_std = self.kwargs["image_obs_noise_std"]

        # Define action space before reset is called
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        self.world = PyBulletWorld(self.has_viewer, self.timestep, self.kwargs)
        self.camera = Camera(self.image_size, self.kwargs)
        self.frame_stack = deque([], maxlen=self.frame_stack_size)

        robot_cfg = self.kwargs["robot"]
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
                policy_input=gym.spaces.Box(  # NEW
                    -np.inf,
                    np.inf,
                    shape=(obs["policy_input"].shape[0],),
                    dtype=np.float32,
                ),
            )
        )

    @property
    def task_reward_function(self):
        """
        Provides the task_reward_function for compatibility with the training script,
        forwarding it from the internal FoldingTask instance.
        """
        return self.task.reward_function

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.current_step = 0
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
        if self.kwargs["dynamics_randomization"]:  # Check if dynamics randomization is good name
            self.world.apply_domain_randomization(self.kwargs)

        # Create robot and cloth
        robot_cfg = self.kwargs["robot"]
        base_pos = robot_cfg["base_pos"]
        base_orn = p.getQuaternionFromEuler(robot_cfg["base_orn_euler"])
        self.robot = PandaRobot(
            base_position=base_pos, base_orientation=base_orn, robot_cfg=robot_cfg
        )
        # Robot dynamics DR (only if master switch is ON)
        if self.kwargs["dynamics_randomization"]:
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

        cloth_cfg = self.kwargs["cloth"]
        initial_cloth_pos_xy = cloth_cfg["initial_pos"]
        cloth_pos = [
            initial_cloth_pos_xy[0],
            initial_cloth_pos_xy[1],
            self.world.get_table_top_z(),
        ]

        # ---- Cloth domain randomization (physics + size + optional color) ----

        # --- Cloth size ---
        if self.kwargs["dynamics_randomization"]:
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
            randomization_kwargs=self.kwargs,
            logger=self.logger,
        )

        # Pass logger down so cloth can report texture DR

        # Wait for cloth to settle
        for _ in range(cloth_cfg["settle_steps"]):
            self.world.step()

        # Set camera target to the cloth's center, matching MuJoCo's lookatbody
        center_v_name = self.cloth.corner_v_names["mid"]
        self._camera_target = self.cloth.get_positions_W()[center_v_name]
        self.camera.begin_episode(self._camera_target)
        # self.camera.print_gui_camera_as_type()
        # (Keep rendering OFF until the end of reset)

        # Initialize the task
        self.task = FoldingTask(
            self.task_name,
            self.cloth,
            self.kwargs,
            self.np_random,
        )

        # Move robot to grasp corner
        corner_v_name = self.cloth.corner_v_names["0"]
        corner_world_pos = self.cloth.get_positions_W()[corner_v_name]

        # First pass IK
        joint_positions = self.robot.calculate_ik(corner_world_pos)
        self.robot.reset_to_joint_positions(joint_positions)

        # Check error and do a second pass only if needed.
        ee_now = self.robot.get_ee_position_W()
        if np.linalg.norm(corner_world_pos - ee_now) > 1e-4:
            joint_positions = self.robot.calculate_ik(corner_world_pos)
            self.robot.reset_to_joint_positions(joint_positions)
            self.world.step()

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
        self.goal, self.goal_noise = self.task.sample_goal(
            self.get_cloth_position_I(), self.cloth.sites
        )

        # Capture initial image (viewer can stay off; camera grabs directly)
        img = self.get_image_obs()
        self.frame_stack.clear()
        for _ in range(self.frame_stack_size):
            self.frame_stack.append(img)

        # Randomize cloth color (skip if DeformableCloth already tinted)
        if self.kwargs["materials_randomization"] and not getattr(
            self.cloth, "_tint_applied", False
        ):
            lo = np.array(cloth_cfg["color_lo"])
            hi = np.array(cloth_cfg["color_hi"])
            self.cloth.set_color((self.np_random.uniform(lo, hi)).tolist())

        # Now show the fully initialized scene (single switch at the very end)
        try:
            if self.has_viewer:
                vcam = self.kwargs["viewer_debug_camera"]
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
                    int(self.kwargs["show_depth_preview"]),
                )
                p.configureDebugVisualizer(
                    p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
                    int(self.kwargs["show_seg_preview"]),
                )
        except Exception:
            pass
        return self.get_obs()

    def step(self, action):
        raw_action = action.copy()
        self.previous_raw_action = raw_action.copy()
        action = raw_action * self.output_max

        # Determine which substep to capture the image on, matching original random logic
        image_obs_substep_idx = int(
            np.clip(
                np.random.normal(
                    self.image_obs_noise_mean * (self.substeps - 1), self.image_obs_noise_std
                ),
                0,
                self.substeps - 1,
            )
        )

        previous_desired_pos_step_W = self.desired_pos_step_W.copy()
        desired_pos_step_W = previous_desired_pos_step_W + action
        self.desired_pos_step_W = np.clip(
            desired_pos_step_W, self.min_absolute_W, self.max_absolute_W
        )

        for i in range(self.substeps):
            for _ in range(self.between_steps):
                self.desired_pos_ctrl_W = (
                    self.filter * self.desired_pos_step_W
                    + (1 - self.filter) * self.desired_pos_ctrl_W
                )

            joint_positions = self.robot.calculate_ik(self.desired_pos_ctrl_W)
            self.robot.apply_joint_positions(joint_positions)
            self.robot.force_fingers_closed()
            self.world.step()

            if i == image_obs_substep_idx:
                self.frame_stack.append(self.get_image_obs())

        obs = self.get_obs()
        reward, done, info = self._get_reward_and_done(obs, raw_action)

        self.current_step += 1

        for k in ("image", "observation", "robot_observation", "achieved_goal", "desired_goal"):
            if np.any(np.isnan(obs[k])):
                raise ValueError(f"NaN in obs['{k}'] detected!")

        return obs, reward, done, info

    def _get_reward_and_done(self, obs, raw_action):
        reward = self.task.compute_reward(obs["achieved_goal"], self.goal, {})

        cloth_pos_I = self.get_cloth_position_I()
        # This is now calculated below using camera projection
        # corner_positions = self._get_corner_image_positions().astype(np.float32)
        # corner_positions = self._get_corner_image_positions()
        # ground-truth corner labels in image space (normalized [0,1])
        corner_positions = self._get_corner_image_positions().astype(np.float32)

        # Calculate all corner distances for the info dict
        # Map vertex name to site name, and site name to its index in the goal vector.
        vertex_to_site_name = {v: s for s, v in self.cloth.sites.items()}
        site_name_to_goal_idx = {c["origin"]: i for i, c in enumerate(self.task.constraints)}

        distances = {}
        all_targets_I = [self.goal[i * 3 : (i + 1) * 3] for i in range(len(self.task.constraints))]

        for corner_key in ("0", "1", "2", "3"):
            v_name = self.cloth.corner_v_names[corner_key]
            achieved_pos_I = cloth_pos_I[v_name]
            site_name = vertex_to_site_name.get(v_name)

            if site_name and site_name in site_name_to_goal_idx:
                # This corner is a constrained "origin" site; calculate distance to its specific target.
                goal_idx = site_name_to_goal_idx[site_name]
                target_pos_I = self.goal[goal_idx * 3 : (goal_idx + 1) * 3]
                distances[corner_key] = float(np.linalg.norm(achieved_pos_I - target_pos_I))
            elif all_targets_I:
                # This corner is not a constrained origin. As a fallback, find its distance
                # to the closest of any of the available targets.
                distances[corner_key] = float(
                    min(np.linalg.norm(achieved_pos_I - t) for t in all_targets_I)
                )
            else:
                # No constraints/targets defined, distance is 0.
                distances[corner_key] = 0.0

        # Use the correct distance for the done condition and success signal
        dist_to_target = distances["1"]
        is_success = dist_to_target < self.success_distance
        # is_success = reward > self.fail_reward

        info = {
            "reward": float(reward),
            "is_success": bool(is_success),
            "delta_size": float(np.linalg.norm(raw_action)),
            "ctrl_error": float(
                np.linalg.norm(self.desired_pos_ctrl_W - self.robot.get_ee_position_W())
            ),
            "corner_sum_error": 0.0,
            "corner_positions": corner_positions.reshape(-1).astype(np.float32),
            "env_memory_usage": self.process.memory_info().rss if self.process else 0,
            "ee_target_W": self.desired_pos_ctrl_W.copy(),
            "ee_target_step_W": self.desired_pos_step_W.copy(),
            "ee_W": self.robot.get_ee_position_W().copy(),
        }

        for k in distances:
            info[f"corner_{k}"] = distances[k]
            info["corner_sum_error"] += distances[k]
        info["dsum"] = info["corner_sum_error"]
        # print(f"Debug: corner_sum_error: {info['corner_sum_error']}")

        if dist_to_target < self.success_distance:
            self.episode_ee_close_steps += 1
        else:
            self.episode_ee_close_steps = 0

        done = self.episode_ee_close_steps >= self.max_close_steps
        # Overwrite is_success to match the original logic where it was tied to reward, not 'done'
        info["is_success"] = bool(is_success)
        # printing info["corner_1"] in the first rollout for each backend.
        # if self.current_step == 1:
        # print(f"Debug: corner_1 distance: {info['corner_1']}")
        return reward, done, info

    def _project_points_uv(self, points_W, label="points", cam_type="default"):
        """
        Project a list of world points with the SAME camera + crop as render_rgb().
        Returns: dict with per-point internals + final normalized UVs.
        """
        # Camera & crop params exactly like render_rgb()
        cam_setup = self.camera.get_stable_camera_setup(self._camera_target, cam_type=cam_type)
        crop = self.camera.get_render_crop_params()
        view, proj = self.camera.get_stable_view_projection_matrices(
            self._camera_target, cam_type=cam_type
        )

        V = np.array(view, dtype=np.float64).reshape(4, 4).T
        P = np.array(proj, dtype=np.float64).reshape(4, 4).T
        VP = P @ V

        W_render, H_render = crop["W_render"], crop["H_render"]
        W_out, H_out = crop["W_out"], crop["H_out"]
        x0, y0 = crop["x0"], crop["y0"]

        per_point = []
        uv_cropped = []
        uv_full = []
        for x, y, z in points_W:
            pos_h = np.array([x, y, z, 1.0], dtype=np.float64)
            clip = VP @ pos_h
            if abs(clip[3]) < 1e-12:
                per_point.append({"world": [float(x), float(y), float(z)], "invalid": True})
                uv_cropped.extend([0.0, 0.0])
                uv_full.extend([0.0, 0.0])
                continue
            ndc = (clip[:3] / clip[3]).astype(np.float64)  # (-1..1)
            x_full = (ndc[0] + 1.0) * 0.5 * W_render
            y_full = (1.0 - ndc[1]) * 0.5 * H_render  # flip Y
            x_crop = x_full - x0
            y_crop = y_full - y0
            u = float(np.clip(x_crop / max(1, W_out), 0.0, 1.0))
            v = float(np.clip(y_crop / max(1, H_out), 0.0, 1.0))
            u_full = float(np.clip(x_full / max(1, W_render), 0.0, 1.0))
            v_full = float(np.clip(y_full / max(1, H_render), 0.0, 1.0))
            per_point.append(
                {
                    "world": [float(x), float(y), float(z)],
                    "clip": [float(c) for c in clip.tolist()],
                    "ndc": [float(n) for n in ndc.tolist()],
                    "pix_full": [float(x_full), float(y_full)],
                    "pix_crop": [float(x_crop), float(y_crop)],
                    "uv_cropped": [u, v],
                    "uv_full": [u_full, v_full],
                }
            )
            uv_cropped.extend([u, v])
            uv_full.extend([u_full, v_full])

        out = {
            "label": label,
            "camera": cam_setup,
            "crop": crop,
            "uv_cropped": uv_cropped,
            "uv_full": uv_full,
            "points": per_point,
        }
        return out

    # ---------- CORNER UVs (kept as the API for drawing) ----------
    def _get_corner_image_positions(self, cam_type="default", space="cropped"):
        """
        Return [u0,v0, u1,v1, u2,v2, u3,v3] in [0,1] for the cloth's corners
        using the *named* cloth corner vertices (parity with MuJoCo).
        """
        pos_dict = self.cloth.get_positions_W()
        names = ["0", "1", "2", "3"]
        named_pts = [pos_dict[self.cloth.corner_v_names[n]] for n in names]
        named_log = self._project_points_uv(named_pts, label="named_corners", cam_type=cam_type)
        key = "uv_full" if space == "full" else "uv_cropped"
        return np.array(named_log[key], dtype=np.float32)

    def get_image_obs(self):
        return self.camera.policy_image(self._camera_target)

    def get_obs(self):
        cloth_pos_I = self.get_cloth_position_I()
        cloth_vel_W = self.cloth.get_velocities_W(self.timestep)

        achieved_goal = self.task.get_achieved_goal(cloth_pos_I, self.cloth.sites)

        cloth_obs = np.concatenate(
            [
                np.array(list(cloth_pos_I.values())).flatten(),
                np.array(list(cloth_vel_W.values())).flatten(),
            ]
        )

        # --- Physics DR scalars (MuJoCo parity) ---
        physics_params = []
        if self.kwargs["dynamics_randomization"]:
            # strict: these must be set during reset()/DR
            g = float(self.world.gravity)
            tab_mu = float(self.world.table_lateral_friction)
            tab_e = float(self.world.table_restitution)
            jd = float(np.mean(self.robot.joint_damping))
            jf = float(np.mean(self.robot.joint_friction))

            physics_params.extend(
                [
                    g,
                    tab_mu,
                    tab_e,
                    float(self.cloth.frictionCoeff),
                    # If these attrs don't exist on your cloth wrapper, drop them or add getters:
                    float(getattr(self.cloth, "thickness", 0.002)),
                    float(getattr(self.cloth, "springElasticStiffness", 40.0)),
                    float(getattr(self.cloth, "springDampingStiffness", 0.1)),
                    jd,
                    jf,
                ]
            )
            cloth_obs = np.concatenate([cloth_obs, np.array(physics_params, dtype=np.float32)])

        ee_pos_W = self.robot.get_ee_position_W()
        ee_vel_W = (ee_pos_W - self._prev_ee_pos_W) / max(self.timestep, 1e-6)
        self._prev_ee_pos_W = ee_pos_W
        ee_pos_I = ee_pos_W - self.relative_origin

        # include desired ctrl in both modes; MuJoCo's "ee" obs has it too
        desired_pos_ctrl_I = self.desired_pos_ctrl_W - self.relative_origin

        if self.robot_observation == "ee":
            robot_obs = np.concatenate([ee_pos_I, ee_vel_W, desired_pos_ctrl_I])
        else:  # "ctrl"
            robot_obs = np.concatenate([self.previous_raw_action, np.zeros(6)])

        image_stack = np.array(list(self.frame_stack)).flatten()
        # concatenated input vector used by the policy: [image | 27-d extras]
        policy_input = np.concatenate([image_stack, self._policy_extra_tail()], axis=0).astype(
            np.float32
        )

        # Sanitize and clip all observation components
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
            "policy_input": policy_input.copy(),  # NEW
        }

    def _policy_extra_tail(self) -> np.ndarray:
        """
        27-D tail to match MuJoCo policy extras.
        Starts from the 9-dim 'ee' block and zero-pads to 27.
        """
        if self.robot_observation == "ee":
            ee_pos_I = self.get_ee_position_I()  # (3,)
            ee_vel_W = self.get_ee_velocity()  # (3,)
            des_I = self.desired_pos_ctrl_W - self.relative_origin  # (3,)
            base = np.concatenate([ee_pos_I, ee_vel_W, des_I]).astype(np.float32)  # (9,)
        else:  # "ctrl"
            base = np.concatenate([self.previous_raw_action, np.zeros(6)], axis=0).astype(
                np.float32
            )
        tail = np.zeros(27, dtype=np.float32)
        tail[: min(27, base.size)] = base[: min(27, base.size)]
        return tail

    def get_ee_position_W(self):
        return self.robot.get_ee_position_W()

    def get_ee_position_I(self):
        return self.robot.get_ee_position_W() - self.relative_origin

    def get_ee_velocity(self):
        ee_pos_W = self.robot.get_ee_position_W()
        ee_vel_W = (ee_pos_W - self._prev_ee_pos_W) / max(self.timestep, 1e-6)
        # Note: _prev_ee_pos_W is already updated in get_obs, which is called every step.
        return ee_vel_W

    def get_joint_positions(self):
        return self.robot.get_joint_positions()

    def get_joint_velocities(self):
        return self.robot.get_joint_velocities()

    def get_cloth_position_I(self):
        """Returns cloth vertex positions relative to the robot's starting grasp point."""
        positions_W = self.cloth.get_positions_W()
        return {k: (v - self.relative_origin) for k, v in positions_W.items()}

    def get_masked_image(
        self,
        point_size=5,
        greyscale=False,
        aux_output=None,
        img=None,
        camera_type="default",
        image_space="cropped",
    ):
        """
        Draw GT (blue) + optional aux (green) corner dots on an image.
        If 'img' is None, a DR-free cropped RGB frame is rendered.
        """
        # DR-free RGB (same as policy view) unless a specific image is provided
        if img is None:
            img = self.camera.render_rgb(self._camera_target, cam_type=camera_type)
        h, w = img.shape[:2]

        # Blue: GT corners in [0,1]
        uv = self._get_corner_image_positions(cam_type=camera_type, space=image_space)
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
        """
        Mirrors MuJoCo's capture_images():
        returns (corner_image, eval_image, cnn_color_image_full, cnn_color_image, cnn_image).
        """
        if aux_output is None:
            print("[Bullet] capture_images: aux_output is None", flush=True)

        # Pre-render a single CROPPED RGB with the DEFAULT stable camera.
        # We'll reuse it for all default-view overlays so the GUI doesn't bounce.
        cropped_rgb_default = self.camera.render_rgb(self._camera_target, cam_type="default")

        # 0) Corner overlay (cropped RGB), larger dots for visibility (8px)
        corner_image = self.get_masked_image(
            point_size=8,
            greyscale=False,
            aux_output=aux_output,
            img=cropped_rgb_default,
            camera_type="default",
            image_space="cropped",
        )

        # 2) CNN color image (FULL render buffer), tiny dots (2px)
        full_rgb = self.camera.render_rgb_full(self._camera_target, cam_type="default")
        cnn_color_image_full = self.get_masked_image(
            point_size=2,
            greyscale=False,
            aux_output=aux_output,
            img=full_rgb,
            camera_type="default",
            image_space="full",
        )

        # 3) CNN color image (cropped) — reuse the same default cropped RGB
        cnn_color_image = self.get_masked_image(
            point_size=2,
            greyscale=False,
            aux_output=aux_output,
            img=cropped_rgb_default,
            camera_type="default",
            image_space="cropped",
        )

        # 4) CNN grayscale image (cropped) — reuse the same default cropped RGB
        cnn_image = self.get_masked_image(
            point_size=2,
            greyscale=True,
            aux_output=aux_output,
            img=cropped_rgb_default,
            camera_type="default",
            image_space="cropped",
        )

        # 1) Eval image from eval_camera (stable), smaller dots (4px), no aux overlay)
        # Do this LAST so the GUI preview (if enabled) settles on the eval view.
        eval_image_rgb = self.camera.render_rgb(self._camera_target, cam_type="eval_camera")
        eval_image = self.get_masked_image(
            point_size=4,
            greyscale=False,
            aux_output=None,
            img=eval_image_rgb,
            camera_type="eval_camera",
            image_space="cropped",
        )

        # If a GUI viewer is active, pin the on-screen debug camera to the eval view.
        if self.has_viewer:
            self.camera.set_debug_camera(self._camera_target, cam_type="eval_camera")

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


class ClothEnvBullet(BulletClothEnv_, EzPickle):
    """Public class to mirror ClothEnv signature and be EzPickle-compatible."""

    def __init__(self, **kwargs):
        BulletClothEnv_.__init__(self, **kwargs)
        EzPickle.__init__(self, **kwargs)
