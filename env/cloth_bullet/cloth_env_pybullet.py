import os
import numpy as np
import gym
from gym.utils import seeding
from collections import deque
from typing import Optional
from multiprocessing import current_process

# Refactored components
from .pybullet_world import PybulletWorld
from .panda_robot import PandaRobot
from .deformable_cloth import DeformableCloth
from .camera import Camera
from .folding_task import FoldingTask

# optional logging
try:
    from df_logging import RunLogger
except Exception:
    RunLogger = None

try:
    import pybullet as p
    import psutil
except Exception as e:
    p = None
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None


class BulletClothEnv_(object):
    """
    PyBullet-Port mit identischer Außen-API zu ClothEnv (MuJoCo).
    This class coordinates the different components of the simulation.
    """
    def __init__(
        self,
        timestep, sparse_dense, success_distance, goal_noise_range, frame_stack_size,
        output_max, success_reward, fail_reward, extra_reward, kp, damping_ratio,
        control_frequency, ctrl_filter, save_folder, randomization_kwargs,
        robot_observation, max_close_steps, model_kwargs_path,
        image_obs_noise_mean=1, image_obs_noise_std=0, has_viewer=False,
        image_size=100, logger: Optional['RunLogger'] = None, **_,
    ):
        if p is None:
            raise ImportError(f"pybullet not available: {_IMPORT_ERR}")

        self._backend_name = "pybullet"
        self.logger = logger
        self.seed()

        # --- Init Params ---
        self.timestep = float(timestep)
        self.control_frequency = float(control_frequency)
        self.substeps = max(1, int(1.0 / (self.timestep * self.control_frequency)))
        self.output_max = float(output_max)
        self.robot_observation = str(robot_observation)
        self.max_close_steps = int(max_close_steps)
        self.success_distance = float(success_distance)
        self.frame_stack_size = int(frame_stack_size)
        self.frame_stack = deque([], maxlen=self.frame_stack_size)
        self.image_size = (int(image_size), int(image_size))
        self.randomization_kwargs = dict(randomization_kwargs or {})
        self.image_obs_noise_mean = image_obs_noise_mean
        self.image_obs_noise_std = image_obs_noise_std

        self.sparse_dense = bool(sparse_dense)
        self.success_reward = float(success_reward)
        self.fail_reward = float(fail_reward)
        self.extra_reward = float(extra_reward)
        self.goal_noise_range = tuple(goal_noise_range)
        self.save_folder = save_folder
        
        # --- UI/Rendering Flags ---
        self._pb_gui = os.getenv("WITH_GUI", "0") == "1"
        self.has_viewer = self._pb_gui and current_process().name == "MainProcess"
        show_full_ui = os.getenv("SHOW_FULL_UI", "0") == "1"
        
        # --- Component Initialization ---
        self.world = PybulletWorld(
            timestep=self.timestep, use_gui=self._pb_gui,
            hide_gui_chrome=not show_full_ui, hide_previews=not show_full_ui
        )
        self.robot: Optional[PandaRobot] = None
        self.cloth: Optional[DeformableCloth] = None
        
        cam_cfg = self.randomization_kwargs.get("camera_config", {})
        cam_fov = float(np.mean(cam_cfg.get("fovy_range", [60.0, 60.0])))
        self.camera = Camera(self.image_size, cam_fov, self.randomization_kwargs)
        if hasattr(self, 'albumentations_transform'):
            self.camera.albumentations_transform = self.albumentations_transform
        
        self.task = FoldingTask(
            task_name="sideways", success_distance=success_distance,
            goal_noise_range=goal_noise_range, sparse_dense=sparse_dense,
            success_reward=success_reward, fail_reward=fail_reward,
            extra_reward=extra_reward, np_random=self.np_random
        )

        # --- Workspace ---
        self.limits_min = [-0.35, -0.35, 0.0]
        self.limits_max = [0.05, 0.05, 0.4]
        
        try:
            self.process = psutil.Process(os.getpid())
        except (NameError, AttributeError):
            self.process = None

        # --- Build simulation and set initial state ---
        self.reset()

        self.desired_pos_ctrl_W = self.relative_origin + np.array([0.05, 0.0, 0.0], np.float32)
        self.desired_pos_step_W = self.desired_pos_ctrl_W.copy()
        # Re-seed the frame stack with the image from this offset state.
        img = self.camera.capture_image(self._fixed_camera_target)
        self.frame_stack.clear()
        for _ in range(self.frame_stack_size):
            self.frame_stack.append(img)

        # --- Action/Observation Spaces ---
        self.action_space = gym.spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
        obs = self.get_obs()
        self.observation_space = gym.spaces.Dict(dict(
            desired_goal=gym.spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype=np.float32),
            achieved_goal=gym.spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype=np.float32),
            observation=gym.spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype=np.float32),
            robot_observation=gym.spaces.Box(-np.inf, np.inf, shape=obs['robot_observation'].shape, dtype=np.float32),
            image=gym.spaces.Box(-np.inf, np.inf, shape=obs['image'].shape, dtype=np.float32),
        ))

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
        self.world.reset()
        
        # Create robot and cloth
        self.robot = PandaRobot(base_position=[0, 0, 0], base_orientation=p.getQuaternionFromEuler([0, 0, 0]))
        cloth_pos = [0.5, 0.0, self.world.get_table_top_z() + 0.05]
        self.cloth = DeformableCloth(base_position=cloth_pos)

        # Let cloth settle
        for _ in range(60): self.world.step()
        
        # Set camera target and update debug view
        self._fixed_camera_target = self.cloth.get_center_W()
        if self.has_viewer:
            # HARDCODE to "default" to match original behavior for the debug camera.
            self.camera.set_debug_camera(self._fixed_camera_target, cam_type="default")
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        # Move robot to grasp corner
        corner_v_name = self.cloth.corner_v_names["0"]
        corner_world_pos = self.cloth.get_positions_W()[corner_v_name]
        
        # First pass
        joint_positions = self.robot.calculate_ik(corner_world_pos)
        self.robot.reset_to_joint_positions(joint_positions)

        # Check error and do a second pass only if needed.
        ee_now = self.robot.get_ee_position_W()
        if np.linalg.norm(corner_world_pos - ee_now) > 1e-4:
            joint_positions = self.robot.calculate_ik(corner_world_pos)
            self.robot.reset_to_joint_positions(joint_positions)
            self.world.step() # Step only after the second pass, if it happens.

        # Anchor cloth to robot
        self.cloth.create_anchor(corner_v_name, self.robot.robot_id, self.robot.hand_link_index)

        # Initialize state variables
        self.relative_origin = self.robot.get_ee_position_W()
        self.desired_pos_step_W = self.relative_origin.copy()
        self.desired_pos_ctrl_W = self.relative_origin.copy()
        self.min_absolute_W = self.relative_origin + np.array(self.limits_min)
        self.max_absolute_W = self.relative_origin + np.array(self.limits_max)
        self.previous_raw_action = np.zeros(3, dtype=np.float32)
        self.episode_ee_close_steps = 0
        self.current_step = 0
        self._prev_ee_pos_W = self.robot.get_ee_position_W()

        # Sample goal and populate frame stack
        self.goal, self.goal_noise = self.task.sample_goal(self.get_cloth_position_I(), self.cloth.sites)
        img = self.camera.capture_image(self._fixed_camera_target)
        self.frame_stack.clear()
        for _ in range(self.frame_stack_size):
            self.frame_stack.append(img)
            
        return self.get_obs()

    def step(self, action):
        raw_action = action.copy()
        action = raw_action * self.output_max
        
        # Determine which substep to capture the image on, matching original random logic
        image_obs_substep_idx = int(np.clip(
            np.random.normal(self.image_obs_noise_mean * (self.substeps - 1), self.image_obs_noise_std),
            0, self.substeps - 1
        ))

        previous_desired_pos_step_W = self.desired_pos_step_W.copy()
        desired_pos_step_W = previous_desired_pos_step_W + action
        self.desired_pos_step_W = np.clip(desired_pos_step_W, self.min_absolute_W, self.max_absolute_W)

        for i in range(self.substeps):
            alpha = (i + 1) / self.substeps
            self.desired_pos_ctrl_W = (1 - alpha) * previous_desired_pos_step_W + alpha * self.desired_pos_step_W
            
            joint_positions = self.robot.calculate_ik(self.desired_pos_step_W)
            self.robot.apply_joint_positions(joint_positions)
            self.robot.force_fingers_closed()
            self.world.step()

            if i == image_obs_substep_idx:
                self.frame_stack.append(self.camera.capture_image(self._fixed_camera_target))
        
        obs = self.get_obs()
        reward, done, info = self._get_reward_and_done(obs, raw_action)
        
        self.previous_raw_action = raw_action.copy()
        self.current_step += 1

        for k in ('image', 'observation', 'robot_observation', 'achieved_goal', 'desired_goal'):
            if np.any(np.isnan(obs[k])):
                raise ValueError(f"NaN in obs['{k}'] detected!")

        return obs, reward, done, info


    def _get_reward_and_done(self, obs, raw_action):
        reward = self.task.compute_reward(obs['achieved_goal'], self.goal, {})
        is_success = reward > self.task.fail_reward

        cloth_pos_I = self.get_cloth_position_I()
        corner_positions = np.array([
            cloth_pos_I[self.cloth.corner_v_names["0"]][:2],
            cloth_pos_I[self.cloth.corner_v_names["1"]][:2],
            cloth_pos_I[self.cloth.corner_v_names["2"]][:2],
            cloth_pos_I[self.cloth.corner_v_names["3"]][:2],
        ], dtype=np.float32)

        info = {
            "reward": float(reward),
            "is_success": bool(is_success),
            "delta_size": float(np.linalg.norm(raw_action)),
            "ctrl_error": float(np.linalg.norm(self.desired_pos_ctrl_W - self.robot.get_ee_position_W())),
            "corner_sum_error": 0.0,
            "corner_positions": corner_positions,
            "env_memory_usage": self.process.memory_info().rss if self.process else 0,
            'ee_target_W': self.desired_pos_ctrl_W.copy(),
            'ee_target_step_W': self.desired_pos_step_W.copy(),
            'ee_W': self.robot.get_ee_position_W().copy(),
        }

        # Calculate all corner distances for the info dict
        inv_map = {v: k for k, v in self.cloth.corner_v_names.items()}
        distances = {"0": 0, "1": 0, "2": 0, "3": 0};
        for c in self.task.constraints:
            s1_v_name = self.cloth.sites.get(c.site1)
            if s1_v_name in inv_map:
                s2_v_name = self.cloth.sites.get(c.site2)
                dist = np.linalg.norm(cloth_pos_I[s1_v_name] - cloth_pos_I[s2_v_name])
                distances[inv_map[s1_v_name]] = dist
        
        for k in distances.keys():
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
        
        cloth_obs = np.concatenate([np.array(list(cloth_pos_I.values())).flatten(), 
                                    np.array(list(cloth_vel_W.values())).flatten()])
        
        ee_pos_W = self.robot.get_ee_position_W()
        ee_vel_W = (ee_pos_W - self._prev_ee_pos_W) / max(self.timestep, 1e-6)
        self._prev_ee_pos_W = ee_pos_W
        ee_pos_I = ee_pos_W - self.relative_origin

        if self.robot_observation == "ee":
            robot_obs = np.concatenate([ee_pos_I, ee_vel_W])
        else: # "ctrl"
            desired_pos_ctrl_I = (self.desired_pos_ctrl_W - self.relative_origin)
            robot_obs = np.concatenate([desired_pos_ctrl_I, ee_vel_W])
            
        image_stack = np.array(list(self.frame_stack)).flatten()
        
        # Sanitize and clip all observation components
        nan = lambda a: np.nan_to_num(a, nan=0.0, posinf=1e3, neginf=-1e3)
        clip = lambda a, mi, ma: np.clip(a, mi, ma)
        return {
            'achieved_goal': clip(nan(achieved_goal), -1e3, 1e3).copy(),
            'desired_goal': clip(nan(self.goal.astype(np.float32)), -1e3, 1e3).copy(),
            'image': clip(np.nan_to_num(image_stack, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).copy(),
            'observation': clip(nan(cloth_obs), -1e3, 1e3).copy().flatten(),
            'robot_observation': clip(nan(robot_obs), -1e3, 1e3).copy().flatten(),
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
            'origin': self.relative_origin,
            'output_max': self.output_max,
            'desired_pos_step_I': self.desired_pos_step_W - self.relative_origin,
            'desired_pos_ctrl_I': self.desired_pos_ctrl_W - self.relative_origin,
            'ee_position_I': self.robot.get_ee_position_W() - self.relative_origin,
            'raw_action': self.previous_raw_action,
            'substeps': self.substeps,
            'timestep': self.timestep,
            'goal_noise': self.goal_noise
        }

class ClothEnvBullet(BulletClothEnv_):
    """Public class to mirror ClothEnv signature."""
    pass