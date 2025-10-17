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
    import psutil
except Exception as e:
    p = None
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None


def _compute_cosine_distance(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b) / denom)


# Constraint-Klasse
class Constraint:
    def __init__(self, site1, site2, distance, noise_directions=None):
        self.site1 = site1
        self.site2 = site2
        self.distance = distance
        self.noise_directions = noise_directions

    def get_achieved_goal(self, cloth_site_positions):
        return cloth_site_positions[self.site1]

    def get_desired_goal(self, cloth_site_positions, noise):
        direction = cloth_site_positions[self.site2] - cloth_site_positions[self.site1]
        if self.noise_directions is not None:
            direction *= np.array(self.noise_directions)
        return cloth_site_positions[self.site1] + direction * noise

    def get_distance(self, cloth_site_positions):
        return np.linalg.norm(cloth_site_positions[self.site1] - cloth_site_positions[self.site2])


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

        self._backend_name = "pybullet"
        self.logger = logger

        # UI/Rendering Flags (über Env steuerbar)
        show_full_ui = os.getenv("SHOW_FULL_UI", "0") == "1"   # zeigt komplette PyBullet-Oberfläche
        self._hide_gui_chrome = not show_full_ui               # Statusleisten/Sliders etc. ausblenden?
        self._hide_previews  = not show_full_ui                # RGB/Depth/Seg-Previews ausblenden?

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
        try:
            self.process = psutil.Process(os.getpid())
        except (NameError, AttributeError):
            self.process = None

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

        # Reward / Tasks
        constraint_infos = _task_definitions.constraints["sideways"](0, 4, 8, self.success_distance)
        self.constraints = [Constraint(
            site1=info['origin'],
            site2=info['target'],
            distance=info['distance'],
            noise_directions=info.get('noise_directions')
        ) for info in constraint_infos]
        self.task_reward_function = _reward_calculation.get_task_reward_function(
            constraint_infos, self.single_goal_dim, self.sparse_dense,
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
        # optional: Ziel/EE ausgeben
        self._print_targets = os.getenv("PRINT_TARGET", "0") == "1"
        self._print_every = int(os.getenv("PRINT_EVERY", "10"))


    # ------------------- Bullet world -------------------
    def _setup_simulation(self):
        """Konfiguriert die Physik-Engine und die Suchpfade."""
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.timestep)
        p.setPhysicsEngineParameter(
            sparseSdfVoxelSize=0.25,
        )

    def _connect_bullet(self):
        if self._pb_gui:
            p.connect(p.GUI)
            if self._hide_gui_chrome:
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            if self._hide_previews:
                p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
                p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
                p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        else:
            p.connect(p.DIRECT)
        # Nur verbinden, keine Simulation hier aufsetzen

    def _build_world(self):
        # Hide rebuild flicker (giant cloth flash) during reset
        _render_was_on = False
        if self._pb_gui:
            _render_was_on = True
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            p.removeAllUserDebugItems()

        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        self._setup_simulation()

        # Lade die Ebene (aus pybullet_data)
        p.loadURDF("plane.urdf")
        self._table_z = 0.0

        # Tisch/Box hinzufügen
        table_half_extents = [0.4, 0.4, 0.02]
        table_pos = [0.5, 0.0, table_half_extents[2]]
        box_collision_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=table_half_extents)
        box_visual_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=table_half_extents, rgbaColor=[0.8, 0.8, 0.8, 1])
        self.table_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=box_collision_shape_id,
                                          baseVisualShapeIndex=box_visual_shape_id, basePosition=table_pos)
        self._table_z = table_pos[2] + table_half_extents[2]  # Oberkante des Tisches

        # Roboter laden (aus pybullet_data)
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)

        # Kleidung laden (aus pybullet_data)
        cloth_pos = [table_pos[0], table_pos[1], self._table_z + 0.05]
        self.cloth_id = p.loadSoftBody(
            "cloth_z_up.obj",
            basePosition=cloth_pos,
            scale=0.15,
            mass=1.0, 
            useNeoHookean=0, 
            useBendingSprings=1, 
            useMassSpring=1,
            springElasticStiffness=40,
            springDampingStiffness=0.1,
            springDampingAllDirections=1,
            useSelfCollision=0,
            frictionCoeff=0.5,
            useFaceContact=1
        )
        p.changeVisualShape(self.cloth_id, -1,
            flags=p.VISUAL_SHAPE_DOUBLE_SIDED, rgbaColor=[0.4, 0.6, 1.0, 1])
        
        # Let the cloth settle to prevent the initial "explosion" flicker
        for _ in range(60):
            p.stepSimulation()

        # Initialize finger link indices before use
        self.left_finger_link_index = None
        self.right_finger_link_index = None
        # Statisches Kameraziel nach dem Settling speichern
        self._fixed_camera_target = self._get_cloth_center_W()

        # cache current verts for our velocity estimate (see step 2)
        mesh = p.getMeshData(self.cloth_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)
        self._prev_soft_verts = np.array(mesh[1], dtype=np.float32)


        # Gelenke & EE
        self.arm_joint_indices = []
        self.ee_link_index = None
        self.grasp_target_link_index = -1 # ID for the grasp point link
        self.finger_joint_indices = []
        self.left_finger_joint_index = None

        # Change visual properties of the gripper
        for j in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, j)
            link_name = info[12].decode('UTF-8')

            # Find revolute joints for arm control
            if info[2] == p.JOINT_REVOLUTE:
                self.arm_joint_indices.append(j)

            # also collect the finger prismatic joints by *joint* name
            if info[2] == p.JOINT_PRISMATIC:
                if 'leftfinger' in link_name:
                    self.finger_joint_indices.append(j)
                    p.setCollisionFilterGroupMask(self.robot_id, j, 1, 0)
                elif 'rightfinger' in link_name:
                    self.finger_joint_indices.append(j)
                    p.setCollisionFilterGroupMask(self.robot_id, j, 1, 0)

            if link_name == 'panda_leftfinger':
                self.left_finger_link_index = j
            elif link_name == 'panda_rightfinger':
                self.right_finger_link_index = j
            elif link_name == 'panda_hand':
                self.ee_link_index = j
                self.hand_link_index = j
            elif link_name == 'panda_grasptarget':
                self.grasp_target_link_index = j
        
        # Finger control params (env-tunable)
        self.finger_closed_pos = 0.0
        self.finger_max_force  = float(os.getenv("FINGER_FORCE", 200))  # much stronger
        self.finger_kp         = float(os.getenv("FINGER_KP", 1.0))
        self.finger_max_vel    = float(os.getenv("FINGER_MAX_VEL", 2.0))

        # after the loop: drive grasp target as EE so control + anchor align
        if self.grasp_target_link_index != -1:
            self.ee_link_index = self.grasp_target_link_index

        if self.ee_link_index is None and self.arm_joint_indices:
            # Fallback to the last link of the arm if 'panda_hand' is not found
            self.ee_link_index = self.arm_joint_indices[-1] + 1

        # Limits/Gains/Forces
        self.joint_lower_limits, self.joint_upper_limits = [], []
        self.joint_max_forces = []
        for j in self.arm_joint_indices:
            ji = p.getJointInfo(self.robot_id, j)
            lo, hi = float(ji[8]), float(ji[9])
            self.joint_lower_limits.append(lo if lo > -1e10 else -3.14)
            self.joint_upper_limits.append(hi if hi <  1e10 else  3.14)
            self.joint_max_forces.append(200.0)

        self.joint_ranges = [u - l for u, l in zip(self.joint_upper_limits, self.joint_lower_limits)]
        self.joint_rest_poses = [0.0] * len(self.arm_joint_indices)

        # neutrale Startpose & Dämpfung
        for j in self.arm_joint_indices:
            p.resetJointState(self.robot_id, j, 0.0, 0.0)
            p.changeDynamics(self.robot_id, j, linearDamping=0.1, angularDamping=0.1)
            # Neutralisiere evtl. Velocity-Controller
            p.setJointMotorControl2(self.robot_id, j, p.VELOCITY_CONTROL, force=0.0)

        for j in self.finger_joint_indices:
            p.resetJointState(self.robot_id, j, self.finger_closed_pos, 0.0)  # start exactly closed
            p.setJointMotorControl2(self.robot_id, j, p.VELOCITY_CONTROL, force=0.0)

        self._prev_ee_pos = np.array(self.get_ee_position_W())
        p.setRealTimeSimulation(0)

        # build 9x9 logical site S0_0..S8_8
        self._compute_cloth_sites(n=9)
        # Identifiziere die Eck-Vertices nach dem Erstellen des Tuchs
        self._build_cloth_sites()

        # Set a stable debug camera and re-enable rendering
        if self._pb_gui and _render_was_on:
            center = self._get_cloth_center_W()
            #cam_type = str(self.randomization_kwargs.get("camera_type", "side")).lower()
            cam_type = "default"
            eye, _ = self._camera_eye_from_type(center, cam_type)  # <-- unpack tuple
            self._set_debug_camera_from_eye(center, eye)  # <-- pass only eye
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        self._weld_fingers_shut()

    def _build_cloth_sites(self):
        # Diese Methode ist jetzt für die Identifizierung der Eck-Vertices zuständig
        mesh = p.getMeshData(self.cloth_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)
        verts = mesh[1]
        min_x = min(v[0] for v in verts); max_x = max(v[0] for v in verts)
        min_y = min(v[1] for v in verts); max_y = max(v[1] for v in verts)

        targets = {
            "top_left":    (min_x, max_y),
            "top_right":   (max_x, max_y),
            "bottom_left": (min_x, min_y),
            "bottom_right":(max_x, min_y),
        }

        def dist2(v, t): return (v[0]-t[0])**2 + (v[1]-t[1])**2

        self.corner_vertex_ids = {}
        for name, t in targets.items():
            self.corner_vertex_ids[name] = min(range(len(verts)), key=lambda i: dist2(verts[i], t))

        # Mapping für Kompatibilität mit bestehendem Code
        self.cloth_site_names = [f"v_{i}" for i in range(len(verts))]
        self.corner_index_mapping = {
            "0": f"v_{self.corner_vertex_ids['top_right']}",
            "1": f"v_{self.corner_vertex_ids['bottom_right']}",
            "2": f"v_{self.corner_vertex_ids['top_left']}",
            "3": f"v_{self.corner_vertex_ids['bottom_left']}",
        }
        self.mid_corner_index = -1 # Nicht mehr anwendbar
        self.max_corner_name = f"v_{self.corner_vertex_ids['bottom_right']}"


    def _weld_fingers_shut(self):
        # Weld each finger link rigidly to the hand link
        for attr in ("_lf_weld", "_rf_weld"):
            cid = getattr(self, attr, None)
            if cid is not None:
                # The simulation is reset, so the constraint ID is invalid.
                # p.resetSimulation() handles cleanup.
                setattr(self, attr, None)
    
        if self.hand_link_index is not None:
            if self.left_finger_link_index is not None:
                self._lf_weld = p.createConstraint(self.robot_id, self.hand_link_index,
                                                   self.robot_id, self.left_finger_link_index,
                                                   jointType=p.JOINT_FIXED, jointAxis=[0,0,0],
                                                   parentFramePosition=[0,0,0], childFramePosition=[0,0,0])
                p.changeConstraint(self._lf_weld, maxForce=self.finger_max_force)
            if self.right_finger_link_index is not None:
                self._rf_weld = p.createConstraint(self.robot_id, self.hand_link_index,
                                                   self.robot_id, self.right_finger_link_index,
                                                   jointType=p.JOINT_FIXED, jointAxis=[0,0,0],
                                                   parentFramePosition=[0,0,0], childFramePosition=[0,0,0])
                p.changeConstraint(self._rf_weld, maxForce=self.finger_max_force)


    def _compute_cloth_sites(self, n=9):
        """Build a stable S{row}_{col} -> vertex index dict from the soft body mesh."""
        mesh = p.getMeshData(self.cloth_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)
        verts = np.array(mesh[1], dtype=np.float32)
        if verts.ndim == 1:
            verts = verts.reshape(-1, 3)

        mins = verts.min(axis=0)
        maxs = verts.max(axis=0)
        xs = np.linspace(mins[0], maxs[0], n)
        ys = np.linspace(mins[1], maxs[1], n)

        sites = {}
        xy = verts[:, :2]
        for r, y in enumerate(ys):
            for c, x in enumerate(xs):
                d2 = (xy[:,0]-x)**2 + (xy[:,1]-y)**2
                idx = int(np.argmin(d2))
                sites[f"S{r}_{c}"] = f"v_{idx}"
        self._cloth_sites_v_indices = sites
        return sites


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
        # _build_world kümmert sich jetzt um den Reset und das Setup
        self._build_world()
        self.reset_osc_values()

        # Pick the corner we want
        cloth_positions_W = self.get_cloth_position_W()
        corner_v_name = self.corner_index_mapping["0"] # e.g., top_right corner
        corner_world = cloth_positions_W[corner_v_name]

        joint_positions = p.calculateInverseKinematics(
            self.robot_id,
            self.ee_link_index,
            corner_world,
            lowerLimits=self.joint_lower_limits,
            upperLimits=self.joint_upper_limits,
            jointRanges=self.joint_ranges,
            restPoses=self.joint_rest_poses,
        )

        # Reset robot joints to the new starting pose
        for i, joint_index in enumerate(self.arm_joint_indices):
            p.resetJointState(self.robot_id, joint_index, joint_positions[i])

        # --- micro-correct: if the link isn't exactly on the corner, nudge once
        ls = p.getLinkState(self.robot_id, self.ee_link_index, computeForwardKinematics=True)
        ee_now = np.array(ls[4])
        delta = corner_world - ee_now
        if np.linalg.norm(delta) > 1e-4:
            joint_positions = p.calculateInverseKinematics(
                self.robot_id, self.ee_link_index, corner_world,
                lowerLimits=self.joint_lower_limits,
                upperLimits=self.joint_upper_limits,
                jointRanges=self.joint_ranges,
                restPoses=self.joint_rest_poses,
            )
            for i, joint_index in enumerate(self.arm_joint_indices):
                p.resetJointState(self.robot_id, joint_index, joint_positions[i])
            p.stepSimulation()

        corner_vertex_index = int(corner_v_name.split('_')[1])
        p.createSoftBodyAnchor(
            self.cloth_id,
            corner_vertex_index,
            self.robot_id,
            self.hand_link_index,
            [0, 0, 0]
        )

        # Update desired positions to the new start
        self.relative_origin = self.get_ee_position_W()
        self.desired_pos_ctrl_W = self.relative_origin.copy()
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
        mesh = p.getMeshData(self.cloth_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)
        verts = mesh[1]
        return {f"v_{i}": np.array(v) for i, v in enumerate(verts)}

    def get_cloth_position_I(self):
        positions_W = self.get_cloth_position_W()
        return {k: (v - self.relative_origin) for k, v in positions_W.items()}

    def get_cloth_velocity(self):
        # Estimate per-vertex velocity from position differences
        mesh = p.getMeshData(self.cloth_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)
        verts = np.array(mesh[1], dtype=np.float32)
        dt = max(self.timestep, 1e-6)
        vels = (verts - getattr(self, "_prev_soft_verts", verts)) / dt
        self._prev_soft_verts = verts
        return {f"v_{i}": vels[i] for i in range(len(vels))}

    # ------------------- camera -------------------
    def _camera_params(self, w=None, h=None):
        """
        MuJoCo-Parität:
        - Lookat = Cloth-Mitte (S4_4)
        - FOV = mean(camera_config['fovy_range'])
        - Eye-Offset abhängig von camera_type ∈ {side, front, up}
        - Optional feinjustierbar via ENV (CAM_*), ohne Codeänderung
        """
        cam_cfg = self.randomization_kwargs.get("camera_config", {}) or {}
        if w is None or h is None:
            w = int(cam_cfg.get("width", self._cam_w))
            h = int(cam_cfg.get("height", self._cam_h))
        self._cam_w, self._cam_h = int(w), int(h)

        fovy_range = cam_cfg.get("fovy_range", [60.0, 60.0])
        fov = float((float(fovy_range[0]) + float(fovy_range[1])) * 0.5)
        self._cam_fov = fov  # 1:1 zu MuJoCo: fov aus Model/Config übernehmen

        # Lookat = Center des Cloth-Grids (S4_4), wie MuJoCo reset_camera() B4_4
        # Statisches Ziel verwenden, damit die Kamera nicht dem Tuch folgt
        center = getattr(self, "_fixed_camera_target", self._get_cloth_center_W())

        # Eye-Offset aus camera_type ableiten (tweakbar via ENV)
        #cam_type = os.getenv("CAM_TYPE", str(self.randomization_kwargs.get("camera_type", "side"))).lower()
        cam_type = "default"
        eye, up = self._camera_eye_from_type(center, cam_type)

        aspect = float(self._cam_w) / max(1.0, float(self._cam_h))
        view = p.computeViewMatrix(eye, center.tolist(), up)
        proj = p.computeProjectionMatrixFOV(self._cam_fov, aspect, 0.01, 2.0)
        return view, proj
    
    def _set_debug_camera_from_eye(self, center, eye):
        """
        Convert (eye, center) into Bullet's (dist, yaw, pitch) and set the GUI camera.
        Env overrides:
        CAM_YAW, CAM_PITCH, CAM_DIST  (degrees / meters)
        """
        import numpy as np, os
        cam_vec = np.array(eye, dtype=np.float32) - np.array(center, dtype=np.float32)  # <-- eye - center
        dist = float(np.linalg.norm(cam_vec) or 0.5)
        # yaw around +Z, pitch negative when looking down from above
        yaw = float(np.degrees(np.arctan2(cam_vec[1], cam_vec[0])))
        pitch = float(-np.degrees(np.arctan2(cam_vec[2], np.linalg.norm(cam_vec[:2]) + 1e-9)))

        # allow manual tweaks
        yaw   = float(os.getenv("CAM_YAW",   yaw))
        pitch = float(os.getenv("CAM_PITCH", pitch))
        dist  = float(os.getenv("CAM_DIST",  dist))

        # keep pitch in a sane range
        pitch = max(-89.0, min(89.0, pitch))

        p.resetDebugVisualizerCamera(dist, yaw, pitch, center.tolist())



    def _get_cloth_center_W(self):
        """Weltkoordinate des Cloth-Mittelpunkts."""
        mesh = p.getMeshData(self.cloth_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)
        verts = np.array(mesh[1])
        return np.mean(verts, axis=0)


    def _camera_eye_from_type(self, center, cam_type):
        """
        Eye = center + Offset; Defaults so gewählt, dass der Ausschnitt MuJoCo ähnlich ist.
        Über ENV kann man live feintunen (Meter):
        CAM_SIDE_DX/DY/DZ, CAM_FRONT_DX/DY/DZ, CAM_UP_DZ etc.
        Returns:
            tuple: (eye_position_list, up_vector_list)
        """
        up = [0.0, 1.0, 0.0]  # Standard "up" vector

        if cam_type == "side":
            dx = float(os.getenv("CAM_SIDE_DX", "-0.55"))
            dy = float(os.getenv("CAM_SIDE_DY", "0.00"))
            dz = float(os.getenv("CAM_SIDE_DZ", "0.30"))
        elif cam_type == "front":
            dx = float(os.getenv("CAM_FRONT_DX", "0.00"))
            dy = float(os.getenv("CAM_FRONT_DY", "-0.65"))
            dz = float(os.getenv("CAM_FRONT_DZ", "0.30"))
        elif cam_type == "up":
            dx = float(os.getenv("CAM_UP_DX", "0.00"))
            dy = float(os.getenv("CAM_UP_DY", "0.00"))
            dz = float(os.getenv("CAM_UP_DZ", "0.80"))
        elif cam_type == "diag":
            dx = float(os.getenv("CAM_DIAG_DX", "1.0"))
            dy = float(os.getenv("CAM_DIAG_DY", "1.0"))
            dz = float(os.getenv("CAM_DIAG_DZ", "1.0"))
            up = [0.0, 0.0, 1.0]
        elif cam_type == "diag_portrait":
            dx = float(os.getenv("CAM_DIAG_DX", "0.4"))
            dy = float(os.getenv("CAM_DIAG_DY", "0.4"))
            dz = float(os.getenv("CAM_DIAG_DZ", "0.4"))
            up = [-1.0, 0.0, 0.0]
        elif cam_type == "side_far":
            dx = float(os.getenv("CAM_SIDE_FAR_DX", "-0.7"))
            dy = float(os.getenv("CAM_SIDE_FAR_DY", "0.0"))
            dz = float(os.getenv("CAM_SIDE_FAR_DZ", "0.4"))
        else:
            up = [0.0, 0.0, 1.0]
            dx = -1.0; dy = -1.0; dz = 1.00


        eye = center + np.array([dx, dy, dz], dtype=np.float32)
        return eye.tolist(), up

    def get_image_obs(self):
        # sicherstellen, dass die Kamera-Parameter aktuell sind
        W, H = self.image_size
        view, proj = self._camera_params(W, H)
        _, _, rgba, _, _ = p.getCameraImage(W, H, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        img = np.reshape(rgba, (H, W, 4))[:, :, :3].astype("uint8")

        # center crop -> image_size
        h0 = int(H / 2 - self.image_size[1] / 2)
        w0 = int(W / 2 - self.image_size[0] / 2)
        h0 = max(0, min(h0, H - self.image_size[1]))
        w0 = max(0, min(w0, W - self.image_size[0]))
        img = img[h0:h0 + self.image_size[1], w0:w0 + self.image_size[0], :]

        try:
            img = self.albumentations_transform(image=img)["image"]
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
        cloth_positions_I = self.get_cloth_position_I()
        for i, constraint in enumerate(self.constraints):
            # Map logical sites (S0_0) to vertex names (v_123)
            site1_v_name = self._cloth_sites_v_indices[constraint.site1]
            site2_v_name = self._cloth_sites_v_indices[constraint.site2]

            # Create a temporary constraint with the correct vertex names
            temp_constraint = Constraint(
                site1=site1_v_name,
                site2=site2_v_name,
                distance=constraint.distance,
                noise_directions=constraint.noise_directions
            )
            goal[i * self.single_goal_dim:(i + 1) * self.single_goal_dim] = \
                (temp_constraint.get_desired_goal(
                    cloth_positions_I, noise)).flatten()
        return goal, noise

    def compute_task_reward(self, achieved_goal, desired_goal, info):
        return self.task_reward_function(achieved_goal, desired_goal, info)

    # ------------------- obs dict -------------------
    def get_obs(self):
        achieved_goal_I = np.zeros(self.single_goal_dim * len(self.constraints), dtype=np.float32)
        cloth_positions_I = self.get_cloth_position_I()
        for i, constraint in enumerate(self.constraints):
            site1_v_name = self._cloth_sites_v_indices[constraint.site1]
            achieved_goal_I[i * self.single_goal_dim:(i + 1) * self.single_goal_dim] = \
                cloth_positions_I[site1_v_name]

        cloth_position = np.array(list(self.get_cloth_position_I().values()), dtype=np.float32)
        cloth_velocity = np.array(list(self.get_cloth_velocity().values()), dtype=np.float32)
        cloth_observation = np.concatenate([cloth_position.flatten(), cloth_velocity.flatten()]).astype(np.float32)

        desired_pos_ctrl_I = (self.desired_pos_ctrl_W - self.relative_origin).astype(np.float32)
        if self.robot_observation == "ee":
            robot_observation = np.concatenate(
                [self.get_ee_position_I(), self.get_ee_velocity()]).astype(np.float32)
        elif self.robot_observation == "ctrl":
            robot_observation = np.concatenate(
                [desired_pos_ctrl_I, self.get_ee_velocity()]).astype(np.float32)
        else:
            raise ValueError(f"unknown robot obs: {self.robot_observation}")

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
            # OSC-Regler
            alpha = (i + 1) / self.substeps
            self.desired_pos_ctrl_W = (1 - alpha) * previous_desired_pos_step_W + alpha * self.desired_pos_step_W
            
            # IK-Ziel für Panda
            joint_positions = p.calculateInverseKinematics(
                self.robot_id,
                self.ee_link_index,
                x_target,
                lowerLimits=self.joint_lower_limits,
                upperLimits=self.joint_upper_limits,
                jointRanges=self.joint_ranges,
                restPoses=self.joint_rest_poses,
            )

            # Gelenk-Steuerung (nur für die 7 Arm-Gelenke)
            p.setJointMotorControlArray(
                self.robot_id,
                self.arm_joint_indices[:7],
                p.POSITION_CONTROL,
                targetPositions=joint_positions[:7],
                forces=self.joint_max_forces[:7],
            )
            # Hard-close the fingers every substep
            for j in self.finger_joint_indices:
                p.setJointMotorControl2(
                    self.robot_id, j, p.POSITION_CONTROL,
                    targetPosition=self.finger_closed_pos,
                    force=self.finger_max_force,
                    positionGain=self.finger_kp,
                    maxVelocity=self.finger_max_vel
                )
                # Safety: if anything drifted, snap it back exactly closed
                js = p.getJointState(self.robot_id, j)[0]
                if abs(js - self.finger_closed_pos) > 1e-5:
                    p.resetJointState(self.robot_id, j, self.finger_closed_pos, 0.0)
            # Physik-Schritt
            p.stepSimulation()

            # Bild-Beobachtung (nur in einem Sub-Schritt)
            if i == image_obs_substep_idx:
                image_obs = self.get_image_obs()
                self.frame_stack.append(image_obs)

            # Debug-Zeug
            if self._pb_gui:
                ee_pos = self.get_ee_position_W()
                ctrl_samples.append(ee_pos)

        obs = self.get_obs()
        reward, done, info = self.post_action(obs, raw_action, cosine_distance)

        self.previous_raw_action = raw_action.copy()
        self.current_step += 1

        # harte NaN-Wache
        for k in ('image', 'observation', 'robot_observation', 'achieved_goal', 'desired_goal'):
            if np.any(np.isnan(obs[k])):
                raise ValueError(f"NaN in obs['{k}'] detected!")

        # für Logs/Debug auch ins info packen
        info['ee_target_W'] = self.desired_pos_ctrl_W.copy()
        info['ee_target_step_W'] = self.desired_pos_step_W.copy()
        info['ee_W'] = self.get_ee_position_W().copy()

        return obs, reward, done, info

    # ------------------- misc -------------------
    def get_corner_constraint_distances(self):
        inv = {v: k for k, v in self.corner_index_mapping.items()}
        distances = {"0": 0, "1": 0, "2": 0, "3": 0}
        cloth_positions_I = self.get_cloth_position_I()
        for i, c in enumerate(self.constraints):
            site1_v_name = self._cloth_sites_v_indices.get(c.site1)
            if site1_v_name in inv:
                site2_v_name = self._cloth_sites_v_indices.get(c.site2)
                dist = np.linalg.norm(cloth_positions_I[site1_v_name] - cloth_positions_I[site2_v_name])
                distances[inv[site1_v_name]] = dist
        return distances

    def post_action(self, obs, raw_action, cosine_distance):
        reward = self.compute_task_reward(np.reshape(obs['achieved_goal'], (1, -1)),
                                          np.reshape(self.goal, (1, -1)), dict())[0]
        is_success = reward > self.fail_reward
        delta_size = float(np.linalg.norm(raw_action))
        ctrl_error = float(np.linalg.norm(self.desired_pos_ctrl_W - self.get_ee_position_W()))
        
        cloth_positions_I = self.get_cloth_position_I()
        corner_positions = np.array([
            cloth_positions_I[self.corner_index_mapping["0"]][:2],
            cloth_positions_I[self.corner_index_mapping["1"]][:2],
            cloth_positions_I[self.corner_index_mapping["2"]][:2],
            cloth_positions_I[self.corner_index_mapping["3"]][:2],
        ], dtype=np.float32)

        info = {
            "reward": float(reward),
            "is_success": bool(is_success),
            "delta_size": delta_size,
            "ctrl_error": ctrl_error,
            "corner_sum_error": 0.0,
            "corner_positions": corner_positions,
            "env_memory_usage": self.process.memory_info().rss if self.process else 0,
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

    def capture_images(self, aux_output=None):
        """
        Erfasst Bilder aus der Simulationskamera, ähnlich der MuJoCo-Implementierung.
        Gibt das Haupt-Kamerabild für alle erwarteten Ausgaben zurück, da
        keine separaten Kameras wie 'eval_camera' konfiguriert sind.
        """
        W, H = self.image_size
        view, proj = self._camera_params(W, H)
        _, _, rgba, _, _ = p.getCameraImage(W, H, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        img = np.reshape(rgba, (H, W, 4))[:, :, :3].astype("uint8")

        # Optional: Zeichne Hilfspunkte, falls `aux_output` gegeben ist.
        # Dies ist eine vereinfachte Darstellung.
        if aux_output is not None and self._pb_gui:
            # aux_output sind normalisierte 2D-Koordinaten.
            # PyBullet hat keine einfache 2D-Overlay-Funktion, daher wird dies übersprungen.
            # Für eine vollständige Implementierung wären Projektionen von 3D-Punkten nötig.
            pass

        # Die Funktion erwartet 5 Bilder, also geben wir 5-mal das erfasste Bild zurück.
        return (
            img.copy(),
            img.copy(),
            img.copy(),
            img.copy(),
            img.copy(),
        )

    def get_trajectory_log_entry(self):
        """
        Sammelt und gibt einen Eintrag für das Trajektorien-Log zurück,
        analog zur MuJoCo-Implementierung.
        """
        entry = {
            'origin': self.relative_origin,
            'output_max': self.output_max,
            'desired_pos_step_I': self.desired_pos_step_W - self.relative_origin,
            'desired_pos_ctrl_I': self.desired_pos_ctrl_W - self.relative_origin,
            'ee_position_I': self.get_ee_position_I(),
            'raw_action': self.previous_raw_action,
            'substeps': self.substeps,
            'timestep': self.timestep,
            'goal_noise': self.goal_noise
        }
        return entry


class ClothEnvBullet(BulletClothEnv_):
    """Public class to mirror ClothEnv signature."""
    pass