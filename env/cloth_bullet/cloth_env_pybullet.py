import contextlib
import os
from collections import deque
from multiprocessing import current_process
from typing import Any, Optional

import gym
import numpy as np
import psutil
from gym.utils import EzPickle, seeding

from env.cloth_bullet.camera import Camera
from env.cloth_bullet.deformable_cloth import DeformableCloth
from env.cloth_bullet.folding_task import FoldingTask
from env.cloth_bullet.panda_robot import PandaRobot

# Refactored components
from env.cloth_bullet.pybullet_world import PyBulletWorld

# NOTE: Do not import mujoco_model_kwargs here.
# Bullet appearance comes solely from randomization_kwargs passed in.


# optional logging
try:
    from df_logging import RunLogger
except Exception:
    RunLogger = None

try:
    import pybullet as p
except Exception as e:
    p = None
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None


class BulletClothEnv_:
    """
    PyBullet-Port mit identischer Außen-API zu ClothEnv (MuJoCo).
    This class coordinates the different components of the simulation.
    """

    def __init__(
        self,
        randomization_kwargs,
        save_folder=None,
        has_viewer=False,
        logger: Optional[Any] = None,
        **_,
    ):
        if _IMPORT_ERR is not None:
            raise _IMPORT_ERR

        self._backend_name = "pybullet"

        if current_process().name != "MainProcess":
            has_viewer = False

        self.process = psutil.Process(os.getpid())
        self.logger = logger
        self.seed()

        # --- Init Params from kwargs ---
        self.kwargs = randomization_kwargs
        self.enable_dr = self.kwargs["enable_dr"]

        task_cfg = self.kwargs["folding_task"]
        self.task_name = self.kwargs["task_name"]
        self.save_folder = save_folder
        self.timestep = float(self.kwargs["timestep"])
        self.control_frequency = float(self.kwargs["control_frequency"])
        self.substeps = max(1, int(1.0 / (self.timestep * self.control_frequency)))
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
        if self.enable_dr:
            self.world.apply_domain_randomization(self.kwargs)

        # Create robot and cloth
        robot_cfg = self.kwargs["robot"]
        base_pos = robot_cfg["base_pos"]
        base_orn = p.getQuaternionFromEuler(robot_cfg["base_orn_euler"])
        self.robot = PandaRobot(
            base_position=base_pos, base_orientation=base_orn, robot_cfg=robot_cfg
        )
        # Robot dynamics DR (only if master switch is ON)
        if self.enable_dr and self.kwargs["dynamics_randomization"]:
            _lin = float(np.random.uniform(*robot_cfg["lin_damping_range"]))
            _ang = float(np.random.uniform(*robot_cfg["ang_damping_range"]))
            _frc = float(np.random.uniform(*robot_cfg["lateral_friction_range"]))
            # Prefer the PandaRobot helper if present; call positionally for max compatibility.
            self.robot.randomize_dynamics(_lin, _ang, _frc)
        else:  # also covers not self.enable_dr
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
        if self.enable_dr:
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
            enable_dr=self.enable_dr,
        )

        # Pass logger down so cloth can report texture DR
        with contextlib.suppress(Exception):
            self.cloth.logger = self.logger

        # Appearance (textures + tint) handled by DeformableCloth
        self.cloth.apply_appearance(self.kwargs)
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
            self.kwargs["folding_task"],
            self.enable_dr,
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
        img = self.camera.capture_image(self._camera_target)
        self.frame_stack.clear()
        for _ in range(self.frame_stack_size):
            self.frame_stack.append(img)

        # Randomize cloth color (skip if DeformableCloth already tinted)
        if (
            self.enable_dr
            and self.kwargs["materials_randomization"]
            and not getattr(self.cloth, "_tint_applied", False)
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

        # --- Lift-then-fold arc motion ---
        robot_cfg = self.kwargs["robot"]
        arc_cfg = robot_cfg["lift_fold_arc"]
        if arc_cfg["enabled"]:
            table_z = self.world.get_table_top_z()
            # t increases as the gripper moves away from its starting XY position
            xy_travel_dist = arc_cfg["xy_travel_dist"]
            t = np.clip(
                np.linalg.norm(self.desired_pos_step_W[:2] - self.relative_origin[:2])
                / xy_travel_dist,
                0.0,
                1.0,
            )
            z_start = table_z + arc_cfg["z_start_offset"]  # Initial height close to the table
            z_end = table_z + arc_cfg["z_end_offset"]  # Peak height of the arc
            self.desired_pos_step_W[2] = (1 - t) * z_start + t * z_end
        # --- End of arc motion logic ---

        for i in range(self.substeps):
            alpha = (i + 1) / self.substeps
            self.desired_pos_ctrl_W = (
                1 - alpha
            ) * previous_desired_pos_step_W + alpha * self.desired_pos_step_W

            joint_positions = self.robot.calculate_ik(self.desired_pos_ctrl_W)
            self.robot.apply_joint_positions(joint_positions)
            self.robot.force_fingers_closed()
            self.world.step()

            if i == image_obs_substep_idx:
                self.frame_stack.append(self.camera.capture_image(self._camera_target))

        obs = self.get_obs()
        reward, done, info = self._get_reward_and_done(obs, raw_action)

        self.previous_raw_action = raw_action.copy()
        self.current_step += 1

        for k in ("image", "observation", "robot_observation", "achieved_goal", "desired_goal"):
            if np.any(np.isnan(obs[k])):
                raise ValueError(f"NaN in obs['{k}'] detected!")

        return obs, reward, done, info

    def _get_reward_and_done(self, obs, raw_action):
        reward = self.task.compute_reward(obs["achieved_goal"], self.goal, {})

        cloth_pos_I = self.get_cloth_position_I()
        corner_positions = np.array(
            [
                cloth_pos_I[self.cloth.corner_v_names["0"]][:2],
                cloth_pos_I[self.cloth.corner_v_names["1"]][:2],
                cloth_pos_I[self.cloth.corner_v_names["2"]][:2],
                cloth_pos_I[self.cloth.corner_v_names["3"]][:2],
            ],
            dtype=np.float32,
        )

        # Calculate all corner distances for the info dict
        inv_map = {v: k for k, v in self.cloth.corner_v_names.items()}
        distances = {"0": 0, "1": 0, "2": 0, "3": 0}
        for c in self.task.constraints:
            s1_v_idx = self.cloth.sites.get(c["origin"])
            if s1_v_idx in inv_map:
                s2_v_idx = self.cloth.sites.get(c["target"])
                dist = np.linalg.norm(cloth_pos_I[s1_v_idx] - cloth_pos_I[s2_v_idx])
                distances[inv_map[s1_v_idx]] = dist

        # Use the correct distance for the done condition and success signal
        dist_to_target = distances.get("1", float("inf"))
        is_success = dist_to_target < self.success_distance

        info = {
            "reward": float(reward),
            "is_success": bool(is_success),
            "delta_size": float(np.linalg.norm(raw_action)),
            "ctrl_error": float(
                np.linalg.norm(self.desired_pos_ctrl_W - self.robot.get_ee_position_W())
            ),
            "corner_sum_error": 0.0,
            "corner_positions": corner_positions,
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
        if self.enable_dr and self.kwargs["dynamics_randomization"]:
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
            robot_obs = np.concatenate([desired_pos_ctrl_I, ee_vel_W])

        image_stack = np.array(list(self.frame_stack)).flatten()

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
        }

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

    def capture_images(self, aux_output=None):
        """Captures an image, returning it 5 times for API compatibility."""
        img = self.camera.capture_image(self._camera_target)
        img = (img.reshape(self.image_size + (-1,)) * 255).astype(np.uint8)
        return (img.copy(), img.copy(), img.copy(), img.copy(), img.copy())

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


def _pick_random_texture(texture_dir):
    """Return a random image path from `texture_dir` or None if not available."""
    try:
        import os
        import random

        if not os.path.isdir(texture_dir):
            return None
        exts = {".png", ".jpg", ".jpeg"}
        candidates = [
            os.path.join(texture_dir, f)
            for f in os.listdir(texture_dir)
            if os.path.splitext(f)[1].lower() in exts
        ]
        if not candidates:
            return None
        return random.choice(candidates)
    except Exception:
        return None


class ClothEnvBullet(BulletClothEnv_, EzPickle):
    """Public class to mirror ClothEnv signature and be EzPickle-compatible."""

    def __init__(self, **kwargs):
        BulletClothEnv_.__init__(self, **kwargs)
        EzPickle.__init__(self, **kwargs)
