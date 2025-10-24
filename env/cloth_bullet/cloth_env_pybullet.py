import contextlib
import os
from collections import deque
from multiprocessing import current_process
from typing import Any, Optional

import albumentations as A
import gym
import numpy as np
import psutil
from gym.utils import seeding

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
        timestep,
        sparse_dense,
        success_distance,
        goal_noise_range,
        frame_stack_size,
        output_max,
        success_reward,
        fail_reward,
        extra_reward,
        control_frequency,
        save_folder,
        randomization_kwargs,
        robot_observation,
        max_close_steps,
        task_name="sideways",
        image_obs_noise_mean=1,
        image_obs_noise_std=0,
        has_viewer=False,
        image_size=100,
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

        # --- Init Params ---
        self.task_name = task_name
        self.save_folder = save_folder
        self.timestep = float(timestep)
        self.control_frequency = float(control_frequency)
        self.substeps = max(1, int(1.0 / (self.timestep * self.control_frequency)))
        self.output_max = float(output_max)
        self.robot_observation = str(robot_observation)
        self.max_close_steps = int(max_close_steps)
        self.success_distance = float(success_distance)
        self.frame_stack_size = int(frame_stack_size)
        self.goal_noise_range = goal_noise_range
        self.sparse_dense = sparse_dense
        self.success_reward = success_reward
        self.fail_reward = fail_reward
        self.extra_reward = extra_reward
        self.image_size = (image_size, image_size)
        self.randomization_kwargs = randomization_kwargs

        # Reuse MuJoCo’s augmentation policy
        self.albumentations_transform = A.Compose(
            [
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Blur(blur_limit=7, p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            ]
        )

        self.has_viewer = has_viewer
        self.image_obs_noise_mean = image_obs_noise_mean
        self.image_obs_noise_std = image_obs_noise_std

        # Define action space before reset is called
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        self.world = PyBulletWorld(self.has_viewer, self.timestep)
        self.camera = Camera(self.image_size, self.randomization_kwargs)
        self.camera.albumentations_transform = self.albumentations_transform
        self.frame_stack = deque([], maxlen=self.frame_stack_size)

        self.limits_min = [-0.35, -0.35, 0.0]
        self.limits_max = [0.35, 0.35, 0.4]

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
        enable_dr = (self.randomization_kwargs or {}).get("enable_dr", True)
        if enable_dr:
            self.world.apply_domain_randomization(self.randomization_kwargs)

        # Create robot and cloth
        self.robot = PandaRobot(
            base_position=[0, 0, 0], base_orientation=p.getQuaternionFromEuler([0, 0, 0])
        )
        # Robot dynamics DR (only if master switch is ON)
        if enable_dr and (self.randomization_kwargs or {}).get("dynamics_randomization", False):
            rob_cfg = (self.randomization_kwargs or {}).get("robot", {})
            _lin = float(np.random.uniform(*rob_cfg.get("lin_damping_range", [0.0, 0.2])))
            _ang = float(np.random.uniform(*rob_cfg.get("ang_damping_range", [0.0, 0.2])))
            _frc = float(np.random.uniform(*rob_cfg.get("lateral_friction_range", [1.5, 3.5])))
            # Prefer the PandaRobot helper if present; call positionally for max compatibility.
            try:
                if hasattr(self.robot, "randomize_dynamics"):
                    self.robot.randomize_dynamics(_lin, _ang, _frc)
                else:
                    raise AttributeError("no helper")
            except (TypeError, AttributeError):
                # Fallback: apply dynamics directly per joint/link
                for j in range(p.getNumJoints(self.robot.robot_id)):
                    p.changeDynamics(
                        self.robot.robot_id,
                        j,
                        linearDamping=_lin,
                        angularDamping=_ang,
                        lateralFriction=_frc,
                    )
        cloth_pos = [0.5, 0.0, self.world.get_table_top_z() + 0.05]

        # ---- Cloth domain randomization (physics + size + optional color) ----
        cloth_cfg = (self.randomization_kwargs or {}).get("cloth", {})

        # --- Cloth size: random if DR enabled, else deterministic ---
        if enable_dr:
            # Priority: cloth.scale_range -> global cloth_size_range -> fallback
            sr = cloth_cfg.get("scale_range", None)
            if sr is None:
                sr = (self.randomization_kwargs or {}).get("cloth_size_range", None)
            if sr is not None:
                lo, hi = float(sr[0]), float(sr[1])
                scale_guess = float(np.random.uniform(lo, hi))
            else:
                scale_guess = float(np.random.uniform(0.20, 0.33))
        else:
            # Deterministic: prefer explicit cloth.scale, else global cloth_size,
            # else a stable default
            scale_guess = float(
                cloth_cfg.get("scale", (self.randomization_kwargs or {}).get("cloth_size", 0.26))
            )

        # Clamp to keep Bullet stable
        scale_guess = float(np.clip(scale_guess, 0.10, 0.38))

        # Spawn a bit higher for larger cloth to avoid initial interpenetration
        base_clearance = 0.05
        extra_clearance = max(0.0, (scale_guess - 0.26)) * 0.35  # gentle slope
        cloth_pos[2] = self.world.get_table_top_z() + base_clearance + extra_clearance
        fr_range = cloth_cfg.get("friction_range", [1.5, 3.5])
        cloth_kwargs = dict(
            scale=scale_guess,
            mass=float(cloth_cfg.get("mass", 0.5)),
            useNeoHookean=int(cloth_cfg.get("useNeoHookean", 0)),
            useBendingSprings=int(cloth_cfg.get("useBendingSprings", 1)),
            useMassSpring=int(cloth_cfg.get("useMassSpring", 1)),
            springElasticStiffness=float(
                np.random.uniform(*cloth_cfg.get("spring_k_range", [30.0, 80.0]))
            ),
            springDampingStiffness=float(
                np.random.uniform(*cloth_cfg.get("spring_c_range", [0.05, 0.2]))
            ),
            springDampingAllDirections=int(cloth_cfg.get("damping_all_dirs", 1)),
            useSelfCollision=int(cloth_cfg.get("useSelfCollision", 1)),
            frictionCoeff=float(np.random.uniform(*fr_range)),
            useFaceContact=int(cloth_cfg.get("useFaceContact", 1)),
        )
        # No target_edge_length yet (DeformableCloth ignores it in current code)
        mesh_path = str(cloth_cfg.get("mesh_path", "cloth_z_up.obj"))
        self.cloth = DeformableCloth(base_position=cloth_pos, mesh_path=mesh_path, **cloth_kwargs)

        # Pass logger down so cloth can report texture DR
        with contextlib.suppress(Exception):
            self.cloth.logger = self.logger

        # Appearance (textures + tint) handled by DeformableCloth
        self.cloth.apply_appearance(self.randomization_kwargs)
        # Wait for cloth to settle
        for _ in range(60):
            self.world.step()

        # Set camera target to MuJoCo's lookatbody, not the table/cloth center
        mujoco_lookatbody = np.array([0.49476399, 0.00668401, 0.13310541], dtype=np.float32)
        self._fixed_camera_target = mujoco_lookatbody
        self.camera.begin_episode(self._fixed_camera_target)
        # (Keep rendering OFF until the end of reset)

        # Initialize the task
        self.task = FoldingTask(
            self.task_name,
            self.cloth,
            self.success_distance,
            self.goal_noise_range,
            self.sparse_dense,
            self.success_reward,
            self.fail_reward,
            self.extra_reward,
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
        img = self.camera.capture_image(self._fixed_camera_target)
        self.frame_stack.clear()
        for _ in range(self.frame_stack_size):
            self.frame_stack.append(img)

        # Randomize cloth color (skip if DeformableCloth already tinted)
        if (
            enable_dr
            and self.randomization_kwargs.get("materials_randomization", False)
            and not getattr(self.cloth, "_tint_applied", False)
        ):
            lo = np.array(cloth_cfg.get("color_lo", [0.3, 0.5, 1.0, 1.0]))
            hi = np.array(cloth_cfg.get("color_hi", [1.0, 1.0, 1.0, 1.0]))
            p.changeVisualShape(
                self.cloth.cloth_id, -1, rgbaColor=(np.random.uniform(lo, hi)).tolist()
            )

        # Now show the fully initialized scene (single switch at the very end)
        try:
            if self.has_viewer:
                p.resetDebugVisualizerCamera(
                    cameraDistance=1.2,
                    cameraYaw=30,
                    cameraPitch=-30,
                    cameraTargetPosition=self._fixed_camera_target,
                )
            # Turn rendering back on…
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            # …and re-enable preview panes (RGB on by default; depth/seg off unless requested)
            if self.has_viewer:
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
                p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
                p.configureDebugVisualizer(
                    p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                    int(self.randomization_kwargs.get("show_depth_preview", 0)),
                )
                p.configureDebugVisualizer(
                    p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
                    int(self.randomization_kwargs.get("show_seg_preview", 0)),
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
        table_z = self.world.get_table_top_z()
        # t increases as the gripper moves away from its starting XY position
        t = np.clip(
            np.linalg.norm(self.desired_pos_step_W[:2] - self.relative_origin[:2]) / 0.25, 0.0, 1.0
        )
        z_start = table_z + 0.03  # Initial height close to the table
        z_end = table_z + 0.10  # Peak height of the arc
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
                self.frame_stack.append(self.camera.capture_image(self._fixed_camera_target))

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
        is_success = reward > self.task.fail_reward

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

        # Calculate all corner distances for the info dict
        inv_map = {v: k for k, v in self.cloth.corner_v_names.items()}
        distances = {"0": 0, "1": 0, "2": 0, "3": 0}
        for c in self.task.constraints:
            s1_v_idx = self.cloth.sites.get(c["origin"])
            if s1_v_idx in inv_map:
                s2_v_idx = self.cloth.sites.get(c["target"])
                dist = np.linalg.norm(cloth_pos_I[s1_v_idx] - cloth_pos_I[s2_v_idx])
                distances[inv_map[s1_v_idx]] = dist

        for k in distances:
            info[f"corner_{k}"] = distances[k]
            info["corner_sum_error"] += distances[k]
        info["dsum"] = info["corner_sum_error"]

        # Use the correct distance for the done condition
        dist_to_target = distances["1"]

        if dist_to_target < self.success_distance:
            self.episode_ee_close_steps += 1
        else:
            self.episode_ee_close_steps = 0

        done = self.episode_ee_close_steps >= self.max_close_steps
        # Overwrite is_success to match the original logic where it was tied to reward, not 'done'
        info["is_success"] = bool(is_success)
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
        if self.randomization_kwargs.get("dynamics_randomization", False):
            # safe fallbacks if something wasn't randomized this episode
            g = float(getattr(self.world, "gravity", -9.81))
            tab_mu = float(getattr(self.world, "table_lateral_friction", 0.8))
            tab_e = float(getattr(self.world, "table_restitution", 0.1))
            jd = float(np.mean(getattr(self.robot, "joint_damping", [0.1])))
            jf = float(np.mean(getattr(self.robot, "joint_friction", [0.8])))

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
        img = self.camera.capture_image(self._fixed_camera_target)
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


class ClothEnvBullet(BulletClothEnv_):
    """Public class to mirror ClothEnv signature."""

    pass
