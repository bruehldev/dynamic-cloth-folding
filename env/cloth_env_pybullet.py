# env/cloth_env_pybullet.py
# PyBullet backend for the cloth environment, API-compatible mit ClothEnv (MuJoCo).
# Minimal bootstrap, genug für Training/Debugging.

import os
import numpy as np
import gym
from gym.utils import seeding
from collections import deque
from typing import Optional
from multiprocessing import current_process

# project utils
from utils import reward_calculation as _reward_calculation
from utils import task_definitions as _task_definitions

# optional logging
import logging
try:
    from df_logging import RunLogger
except Exception:
    RunLogger = None

try:
    import pybullet as p
    import pybullet_data
except Exception as e:
    p = None
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None


def _compute_cosine_distance(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b) / denom)


class BulletClothEnv_(object):
    """
    PyBullet-Port mit identischer Außen-API zu ClothEnv (MuJoCo):
    - action_space: Box(3,) (Delta in Meter)
    - observation_space: Dict mit keys ['image','observation','robot_observation','achieved_goal','desired_goal']
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
        kp,
        damping_ratio,
        control_frequency,
        ctrl_filter,
        save_folder,
        randomization_kwargs,
        robot_observation,
        max_close_steps,
        model_kwargs_path,   # unbenutzt im Bullet-Bootstrap
        image_obs_noise_mean=1,
        image_obs_noise_std=0,
        has_viewer=False,
        image_size=100,
        logger: Optional['RunLogger'] = None,
        **_,
    ):
        if p is None:
            raise ImportError(f"pybullet not available: {_IMPORT_ERR}")

        # --- who am I?
        self._backend_name = "pybullet"
        self.logger = logger
        # GUI nur im Hauptprozess und nur wenn explizit via Env-Var WITH_GUI=1 angefordert.
        # Diese Env-Var hat Vorrang vor der `has_viewer`-Einstellung in der Config.
        self._pb_gui = os.getenv("WITH_GUI", "0") == "1" and current_process().name == "MainProcess"
        self.has_viewer = self._pb_gui

        # --- params
        self.timestep = float(timestep)
        self.control_frequency = float(control_frequency)
        self.substeps = max(1, int(1.0 / (self.timestep * self.control_frequency)))
        self.filter = float(ctrl_filter)
        self.success_distance = float(success_distance)
        self.goal_noise_range = tuple(goal_noise_range)
        self.sparse_dense = bool(sparse_dense)
        self.success_reward = float(success_reward)
        self.fail_reward = float(fail_reward)
        self.extra_reward = float(extra_reward)
        self.output_max = float(output_max)
        self.kp = float(kp)
        self.damping_ratio = float(damping_ratio)
        self.randomization_kwargs = dict(randomization_kwargs or {})
        self.robot_observation = str(robot_observation)
        self.max_close_steps = int(max_close_steps)
        self.image_size = (int(image_size), int(image_size))
        self.save_folder = save_folder
        self.image_obs_noise_mean = image_obs_noise_mean
        self.image_obs_noise_std = image_obs_noise_std
        self.frame_stack_size = int(frame_stack_size)
        self.frame_stack = deque([], maxlen=self.frame_stack_size)

        self.single_goal_dim = 3
        self.process = None  # optional psutil

        # workspace (relativ zur Reset-EE-Position)
        self.limits_min = [-0.35, -0.35, 0.0]
        self.limits_max = [0.05, 0.05, 0.4]

        # Kamera
        cam_cfg = self.randomization_kwargs.get("camera_config", {})
        self._cam_w = int(cam_cfg.get("width", 160))
        self._cam_h = int(cam_cfg.get("height", 120))
        self._cam_fov = float(np.mean(cam_cfg.get("fovy_range", [60.0, 60.0])))

        # Welt + Roboter
        self._connect_bullet()
        self._build_world()

        # Start
        self.relative_origin = self.get_ee_position_W()
        self.desired_pos_ctrl_W = self.relative_origin + np.array([0.05, 0.0, 0.0], np.float32)
        self.desired_pos_step_W = self.desired_pos_ctrl_W.copy()
        self.min_absolute_W = self.relative_origin + np.array(self.limits_min)
        self.max_absolute_W = self.relative_origin + np.array(self.limits_max)

        # Action/Obs Spaces
        self.action_space = gym.spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
        self._build_cloth_sites()

        # Reward / Tasks
        self.constraints = _task_definitions.constraints["sideways"](0, 4, 8, self.success_distance)
        self.task_reward_function = _reward_calculation.get_task_reward_function(
            self.constraints, self.single_goal_dim, self.sparse_dense,
            self.success_reward, self.fail_reward, self.extra_reward
        )

        # Buchhaltung / Reset-Frames
        self.seed()
        self.reset_osc_values()
        self.goal, self.goal_noise = self.sample_goal_I()

        img = self.get_image_obs()
        img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
        for _ in range(self.frame_stack_size):
            self.frame_stack.append(img)

        obs = self.get_obs()
        self.observation_space = gym.spaces.Dict(dict(
            desired_goal=gym.spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype=np.float32),
            achieved_goal=gym.spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype=np.float32),
            observation=gym.spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype=np.float32),
            robot_observation=gym.spaces.Box(-np.inf, np.inf, shape=obs['robot_observation'].shape, dtype=np.float32),
            image=gym.spaces.Box(-np.inf, np.inf, shape=obs['image'].shape, dtype=np.float32),
        ))

    # ------------------- Bullet world -------------------
    def _connect_bullet(self):
        if self._pb_gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(self.timestep)
        p.setGravity(0, 0, -9.81)

    def _build_world(self):
        p.loadURDF("plane.urdf")
        self._table_z = 0.0
        self.robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)

        # Gelenke & EE
        self.arm_joint_indices = []
        self.ee_link_index = None
        for j in range(p.getNumJoints(self.robot_id)):
            ji = p.getJointInfo(self.robot_id, j)
            if ji[2] == p.JOINT_REVOLUTE:
                self.arm_joint_indices.append(j)
            name = ji[12].decode() if isinstance(ji[12], (bytes, bytearray)) else str(ji[12])
            if name.endswith("link_7"):
                self.ee_link_index = j
        if self.ee_link_index is None and self.arm_joint_indices:
            self.ee_link_index = self.arm_joint_indices[-1]

        # Limits/Gains/Forces
        self.joint_lower_limits, self.joint_upper_limits = [], []
        self.joint_max_forces = []
        for j in self.arm_joint_indices:
            ji = p.getJointInfo(self.robot_id, j)
            lo, hi = float(ji[8]), float(ji[9])
            self.joint_lower_limits.append(lo if lo > -1e10 else -3.14)
            self.joint_upper_limits.append(hi if hi <  1e10 else  3.14)
            self.joint_max_forces.append(200.0)

        # neutrale Startpose & Dämpfung
        for j in self.arm_joint_indices:
            p.resetJointState(self.robot_id, j, 0.0, 0.0)
            p.changeDynamics(self.robot_id, j, linearDamping=0.04, angularDamping=0.04)
            # Neutralisiere evtl. Velocity-Controller
            p.setJointMotorControl2(self.robot_id, j, p.VELOCITY_CONTROL, force=0.0)

        self._prev_ee_pos = np.array(self.get_ee_position_W())
        p.setRealTimeSimulation(0)

    def _build_cloth_sites(self):
        cloth_size = float(self.randomization_kwargs.get('cloth_size', 0.24))
        grid_n = 9
        half = cloth_size / 2.0
        xs = np.linspace(-half, half, grid_n)
        ys = np.linspace(-half, half, grid_n)
        z = self._table_z + 0.0
        self._cloth_sites_W = {}
        self.corner_index_mapping = {"0": f"S0_{grid_n-1}", "1": f"S{grid_n-1}_{grid_n-1}",
                                     "2": "S0_0", "3": f"S{grid_n-1}_0"}
        self.cloth_site_names = []
        for i, xv in enumerate(xs):
            for j, yv in enumerate(ys):
                name = f"S{i}_{j}"
                self._cloth_sites_W[name] = np.array([0.6 + xv, yv, z])
                if i in (0, 4, grid_n-1) and j in (0, 4, grid_n-1):
                    self.cloth_site_names.append(name)
        self.mid_corner_index = 4
        self.max_corner_name = f"S{grid_n-1}_{grid_n-1}"

    # ------------------- Gym hooks -------------------
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.seed_val = seed
        return [seed]

    def reset_osc_values(self):
        self.previous_raw_action = np.zeros(3, dtype=np.float32)
        self.raw_action = None
        self.episode_ee_close_steps = 0
        self.current_step = 0

    def reset(self):
        p.resetSimulation()
        self._build_world()
        self.reset_osc_values()
        self.relative_origin = self.get_ee_position_W()
        self.desired_pos_ctrl_W = self.relative_origin + np.array([0.05, 0.0, 0.0], np.float32)
        self.desired_pos_step_W = self.desired_pos_ctrl_W.copy()
        self.min_absolute_W = self.relative_origin + np.array(self.limits_min)
        self.max_absolute_W = self.relative_origin + np.array(self.limits_max)
        self.goal, self.goal_noise = self.sample_goal_I()

        img = self.get_image_obs()
        img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
        self.frame_stack.clear()
        for _ in range(self.frame_stack_size):
            self.frame_stack.append(img)
        return self.get_obs()

    # ------------------- sensors -------------------
    def get_ee_position_W(self):
        ls = p.getLinkState(self.robot_id, self.ee_link_index, computeForwardKinematics=True)
        return np.array(ls[4])

    def get_ee_position_I(self):
        return self.get_ee_position_W() - self.relative_origin

    def get_ee_velocity(self):
        pos = self.get_ee_position_W()
        vel = (pos - self._prev_ee_pos) / max(self.timestep, 1e-6)
        self._prev_ee_pos = pos
        return vel

    def get_joint_positions(self):
        return np.array([p.getJointState(self.robot_id, j)[0] for j in self.arm_joint_indices])

    def get_joint_velocities(self):
        return np.array([p.getJointState(self.robot_id, j)[1] for j in self.arm_joint_indices])

    def get_cloth_position_W(self):
        return {k: v.copy() for k, v in self._cloth_sites_W.items()}

    def get_cloth_position_I(self):
        return {k: (v - self.relative_origin) for k, v in self._cloth_sites_W.items()}

    def get_cloth_velocity(self):
        return {k: np.zeros(3) for k in self._cloth_sites_W.keys()}

    # ------------------- camera -------------------
    def _camera_params(self):
        target = np.array([0.6, 0.0, self._table_z])
        eye = target + np.array([0.0, 0.0, 0.8])
        up = [0, 1, 0]
        aspect = float(self._cam_w) / float(self._cam_h)
        view = p.computeViewMatrix(eye, target, up)
        proj = p.computeProjectionMatrixFOV(self._cam_fov, aspect, 0.01, 2.0)
        return view, proj

    def get_image_obs(self):
        import cv2
        W, H = self._cam_w, self._cam_h
        view, proj = self._camera_params()
        _, _, rgba, _, _ = p.getCameraImage(W, H, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        img = np.reshape(rgba, (H, W, 4))[:, :, :3].astype("uint8")

        # center crop -> image_size
        h0 = int(H / 2 - self.image_size[1] / 2)
        w0 = int(W / 2 - self.image_size[0] / 2)
        h0 = max(0, min(h0, H - self.image_size[1]))
        w0 = max(0, min(w0, W - self.image_size[0]))
        img = img[h0:h0 + self.image_size[1], w0:w0 + self.image_size[0], :]

        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except Exception:
            pass

        img = img.astype(np.float32) / 255.0
        img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
        img = np.clip(img, 0.0, 1.0)
        return img.flatten().copy()

    # ------------------- goal/reward -------------------
    def sample_goal_I(self):
        goal = np.zeros(self.single_goal_dim * len(self.constraints), dtype=np.float32)
        noise = self.np_random.uniform(self.goal_noise_range[0], self.goal_noise_range[1])
        for i, constraint in enumerate(self.constraints):
            target = constraint['target']
            target_pos = self.get_cloth_position_W()[target].copy()
            offset = np.zeros(self.single_goal_dim, dtype=np.float32)
            if 'noise_directions' in constraint:
                for idx, offset_dir in enumerate(constraint['noise_directions']):
                    offset[idx] = offset_dir * noise
            goal[i*self.single_goal_dim:(i+1)*self.single_goal_dim] = target_pos + offset - self.relative_origin
        return goal.copy(), noise

    def compute_task_reward(self, achieved_goal, desired_goal, info):
        return self.task_reward_function(achieved_goal, desired_goal, info)

    # ------------------- obs dict -------------------
    def get_obs(self):
        achieved_goal_I = np.zeros(self.single_goal_dim * len(self.constraints), dtype=np.float32)
        for i, constraint in enumerate(self.constraints):
            origin = constraint['origin']
            p_W = self.get_cloth_position_W()[origin].copy()
            achieved_goal_I[i*self.single_goal_dim:(i+1)*self.single_goal_dim] = (p_W - self.relative_origin).astype(np.float32)

        cloth_position = np.array(list(self.get_cloth_position_I().values()), dtype=np.float32)
        cloth_velocity = np.array(list(self.get_cloth_velocity().values()), dtype=np.float32)
        cloth_observation = np.concatenate([cloth_position.flatten(), cloth_velocity.flatten()]).astype(np.float32)

        desired_pos_ctrl_I = (self.desired_pos_ctrl_W - self.relative_origin).astype(np.float32)
        if self.robot_observation == "ee":
            robot_observation = np.concatenate([self.get_ee_position_I().astype(np.float32),
                                                self.get_ee_velocity().astype(np.float32),
                                                desired_pos_ctrl_I]).astype(np.float32)
        elif self.robot_observation == "ctrl":
            robot_observation = np.concatenate([self.previous_raw_action.astype(np.float32),
                                                np.zeros(6, dtype=np.float32)]).astype(np.float32)
        else:
            robot_observation = np.zeros(9, dtype=np.float32)

        image_stack = np.array([img for img in self.frame_stack], dtype=np.float32).flatten()

        # sanitize & clamp
        nan = lambda a: np.nan_to_num(a, nan=0.0, posinf=1e3, neginf=-1e3)
        achieved_goal_I = nan(achieved_goal_I)
        goal = nan(self.goal.astype(np.float32))
        cloth_observation = nan(cloth_observation)
        robot_observation = nan(robot_observation)
        image_stack = np.nan_to_num(image_stack, nan=0.0, posinf=1.0, neginf=0.0)

        cloth_observation = np.clip(cloth_observation, -1e3, 1e3)
        robot_observation = np.clip(robot_observation, -1e3, 1e3)
        achieved_goal_I = np.clip(achieved_goal_I, -1e3, 1e3)
        goal = np.clip(goal, -1e3, 1e3)
        image_stack = np.clip(image_stack, 0.0, 1.0)

        return {
            'achieved_goal': achieved_goal_I.copy(),
            'desired_goal': goal.copy(),
            'image': image_stack.copy(),
            'observation': cloth_observation.copy().flatten(),
            'robot_observation': robot_observation.copy().flatten(),
        }

    # ------------------- step -------------------
    def step(self, action):
        raw_action = action.copy()
        action = raw_action * self.output_max

        prev_action_before_update = self.previous_raw_action.copy()
        image_obs_substep_idx = int(
            np.clip(
                np.random.normal(self.image_obs_noise_mean * (self.substeps - 1),
                                 self.image_obs_noise_std),
                0, self.substeps - 1
            )
        )
        cosine_distance = _compute_cosine_distance(self.previous_raw_action, raw_action)

        previous_desired_pos_step_W = self.desired_pos_step_W.copy()
        desired_pos_step_W = previous_desired_pos_step_W + action
        self.desired_pos_step_W = np.clip(desired_pos_step_W, self.min_absolute_W, self.max_absolute_W)
        x_target = self.desired_pos_step_W.copy()

        ctrl_samples = []
        image_obs = None
        self.current_step = getattr(self, "current_step", 0)

        for i in range(self.substeps):
            # low-pass
            self.desired_pos_ctrl_W = self.filter * self.desired_pos_step_W + (1 - self.filter) * self.desired_pos_ctrl_W

            # (optional) Linie EE -> Ziel
            if self._pb_gui:
                p.addUserDebugLine(self.get_ee_position_W(),
                                   self.desired_pos_ctrl_W, [1, 0, 0], 2, lifeTime=0.1)

            # IK
            cur = p.getLinkState(self.robot_id, self.ee_link_index)
            target_orn = cur[5]
            q_full = p.calculateInverseKinematics(
                self.robot_id, self.ee_link_index,
                self.desired_pos_ctrl_W.tolist(), target_orn,
                lowerLimits=self.joint_lower_limits,
                upperLimits=self.joint_upper_limits,
                jointRanges=[(u - l) for (l, u) in zip(self.joint_lower_limits, self.joint_upper_limits)],
                restPoses=[p.getJointState(self.robot_id, j)[0] for j in self.arm_joint_indices],
                maxNumIterations=100, residualThreshold=1e-4
            )
            qpos = list(q_full[:len(self.arm_joint_indices)])
            # clamp
            for k, (lo, hi) in enumerate(zip(self.joint_lower_limits, self.joint_upper_limits)):
                qpos[k] = min(max(qpos[k], lo), hi)

            # kräftig antreiben (robust per joint)
            for j, q in zip(self.arm_joint_indices, qpos):
                p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL,
                                        targetPosition=float(q),
                                        positionGain=0.8, velocityGain=1.0, force=200.0)

            # mehrere Integrationsschritte
            for _ in range(5):
                p.stepSimulation()

            if i in (0, int(self.substeps / 2), int(self.substeps - 1)):
                ctrl_samples.append({
                    "idx": int(i + 1),
                    "x_des": self.desired_pos_ctrl_W.tolist(),
                    "q": self.get_joint_positions().tolist(),
                    "dq": self.get_joint_velocities().tolist(),
                    "x_ee": self.get_ee_position_W().tolist(),
                })

            if i == image_obs_substep_idx:
                image_obs = self.get_image_obs()
                self.frame_stack.append(image_obs)

        obs = self.get_obs()
        reward, done, info = self.post_action(obs, raw_action, cosine_distance)

        # TEMP: fake corner labels, damit Aux-Loss kein NaN erzeugt
        info['corner_positions'] = np.array([[0.1, 0.1],
                                             [0.9, 0.1],
                                             [0.1, 0.9],
                                             [0.9, 0.9]], dtype=np.float32)

        self.previous_raw_action = raw_action.copy()
        self.current_step += 1

        # harte NaN-Wache
        for k in ('image', 'observation', 'robot_observation', 'achieved_goal', 'desired_goal'):
            arr = np.asarray(obs[k])
            if not np.all(np.isfinite(arr)):
                raise RuntimeError(f"NaN/Inf detected in obs['{k}'] at step {self.current_step}")

        return obs, reward, done, info

    # ------------------- misc -------------------
    def get_corner_constraint_distances(self):
        inv = {v: k for k, v in self.corner_index_mapping.items()}
        distances = {"0": 0, "1": 0, "2": 0, "3": 0}
        for i, c in enumerate(self.constraints):
            if c['origin'] in inv:
                origin_pos = self.get_cloth_position_W()[c['origin']].copy() - self.relative_origin
                target_pos = self.goal[i*self.single_goal_dim:(i+1)*self.single_goal_dim]
                distances[inv[c['origin']]] = float(np.linalg.norm(origin_pos - target_pos))
        return distances

    def post_action(self, obs, raw_action, cosine_distance):
        reward = self.compute_task_reward(np.reshape(obs['achieved_goal'], (1, -1)),
                                          np.reshape(self.goal, (1, -1)), dict())[0]
        is_success = reward > self.fail_reward
        delta_size = float(np.linalg.norm(raw_action))
        ctrl_error = float(np.linalg.norm(self.desired_pos_ctrl_W - self.get_ee_position_W()))
        info = {
            "reward": float(reward),
            "is_success": bool(is_success),
            "delta_size": delta_size,
            "ctrl_error": ctrl_error,
            "corner_sum_error": 0.0,
        }
        dists = self.get_corner_constraint_distances()
        for k in dists.keys():
            info[f"corner_{k}"] = dists[k]
            info["corner_sum_error"] += dists[k]
        info["dsum"] = info["corner_sum_error"]
        done = False
        if dists["1"] < self.success_distance:
            self.episode_ee_close_steps += 1
        else:
            self.episode_ee_close_steps = 0
        if self.episode_ee_close_steps >= self.max_close_steps:
            done = True
        return reward, done, info


class ClothEnvBullet(BulletClothEnv_):
    """Public class to mirror ClothEnv signature."""
    pass
