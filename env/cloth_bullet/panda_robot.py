import numpy as np
import pybullet as p


class PandaRobot:
    """
    Encapsulates the Franka Emika Panda robot logic, including loading,
    joint/link identification, control, and sensor feedback.
    """

    def __init__(self, base_position, base_orientation, robot_cfg=None):
        self.urdf_path = robot_cfg["urdf_path"]
        self.robot_id = p.loadURDF(
            self.urdf_path, base_position, base_orientation, useFixedBase=True
        )

        self._find_links_and_joints()
        self._get_joint_limits()
        self._setup_finger_control(robot_cfg["finger"])
        self._setup_ik_params(robot_cfg.get("ik", {}))

        # --- NEW: explicit joint-space position controller gains ---
        arm_ctrl = (robot_cfg or {}).get("arm_control", {})
        self.arm_pos_gain = float(arm_ctrl.get("position_gain", 0.25))
        self.arm_vel_gain = float(arm_ctrl.get("velocity_gain", 1.0))
        self.arm_max_force_scale = float(arm_ctrl.get("max_force_scale", 1.0))
        self.set_initial_joint_positions(robot_cfg["init_joint_positions"])
        self.weld_fingers_shut()

    def _find_links_and_joints(self):
        """Finds and stores important link and joint indices from the URDF."""
        self.arm_joint_indices = []
        self.finger_joint_indices = []
        self.ee_link_index = None
        self.hand_link_index = None
        self.left_finger_link_index = None
        self.right_finger_link_index = None
        self.grasp_target_link_index = -1

        for j in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, j)
            link_name = info[12].decode("UTF-8")
            joint_name = info[1].decode("UTF-8")

            if info[2] == p.JOINT_REVOLUTE:
                self.arm_joint_indices.append(j)
            elif info[2] == p.JOINT_PRISMATIC and "finger" in joint_name:
                self.finger_joint_indices.append(j)
                p.setCollisionFilterGroupMask(self.robot_id, j, 1, 0)

            if link_name == "panda_leftfinger":
                self.left_finger_link_index = j
            elif link_name == "panda_rightfinger":
                self.right_finger_link_index = j
            elif link_name == "panda_hand":
                self.hand_link_index = j
            elif link_name == "panda_grasptarget":
                self.grasp_target_link_index = j

        # Use the grasp target as the primary end-effector for control
        if self.grasp_target_link_index != -1:
            self.ee_link_index = self.grasp_target_link_index
        elif self.hand_link_index is not None:
            self.ee_link_index = self.hand_link_index
        elif self.arm_joint_indices:
            self.ee_link_index = self.arm_joint_indices[-1] + 1

    def _get_joint_limits(self):
        """
        Queries and stores the joint limits (range, rest poses) for the arm.
        """
        self.joint_limits_lower = []
        self.joint_limits_upper = []
        self.joint_ranges = []
        self.joint_rest_poses = []
        self.joint_max_forces = []

        for i in self.arm_joint_indices:
            info = p.getJointInfo(self.robot_id, i)
            self.joint_limits_lower.append(info[8])
            self.joint_limits_upper.append(info[9])
            self.joint_ranges.append(info[9] - info[8])
            self.joint_max_forces.append(info[10])
            # Simple midpoint rest pose
            self.joint_rest_poses.append((info[8] + info[9]) / 2)

    def _setup_finger_control(self, finger_cfg=None):
        """
        Sets up motor control for the robot's fingers and welds them shut.
        """
        self.finger_closed_pos = float(finger_cfg["closed_pos"])
        self.finger_max_force = float(finger_cfg["max_force"])
        self.finger_kp = float(finger_cfg["kp"])
        self.finger_max_vel = float(finger_cfg["max_vel"])

    def _setup_ik_params(self, ik_cfg=None):
        ik_cfg = ik_cfg or {}
        self.ik_max_iters = int(ik_cfg.get("max_iters", 100))
        self.ik_residual_threshold = float(ik_cfg.get("residual_threshold", 1e-4))
        # NEW: orientation-aware IK options
        self.ik_use_orientation = bool(ik_cfg.get("use_orientation", True))
        if "target_quat_xyzw" in ik_cfg:
            self._ik_target_quat = tuple(map(float, ik_cfg["target_quat_xyzw"]))
        elif "target_euler_rpy" in ik_cfg:
            self._ik_target_quat = p.getQuaternionFromEuler(
                [float(x) for x in ik_cfg["target_euler_rpy"]]
            )
        else:
            # Default “tool-down” wrt world: rotate 180° about Y
            self._ik_target_quat = p.getQuaternionFromEuler([0.0, np.pi, 0.0])

    def weld_fingers_shut(self):
        """Creates fixed constraints to weld the fingers to the hand, ensuring a rigid grip."""
        if self.hand_link_index is not None:
            if self.left_finger_link_index is not None:
                cid = p.createConstraint(
                    self.robot_id,
                    self.hand_link_index,
                    self.robot_id,
                    self.left_finger_link_index,
                    jointType=p.JOINT_FIXED,
                    jointAxis=[0, 0, 0],
                    parentFramePosition=[0, 0, 0],
                    childFramePosition=[0, 0, 0],
                )
                p.changeConstraint(cid, maxForce=self.finger_max_force)
            if self.right_finger_link_index is not None:
                cid = p.createConstraint(
                    self.robot_id,
                    self.hand_link_index,
                    self.robot_id,
                    self.right_finger_link_index,
                    jointType=p.JOINT_FIXED,
                    jointAxis=[0, 0, 0],
                    parentFramePosition=[0, 0, 0],
                    childFramePosition=[0, 0, 0],
                )
                p.changeConstraint(cid, maxForce=self.finger_max_force)

    def reset_to_neutral(self):
        """
        Resets the robot's arm joints to a neutral pose (all zeros).
        """
        for j in self.arm_joint_indices:
            p.resetJointState(self.robot_id, j, 0.0, 0.0)
            p.setJointMotorControl2(self.robot_id, j, p.VELOCITY_CONTROL, force=0.0)
        for j in self.finger_joint_indices:
            p.resetJointState(self.robot_id, j, self.finger_closed_pos, 0.0)
            p.setJointMotorControl2(self.robot_id, j, p.VELOCITY_CONTROL, force=0.0)

    def reset_to_joint_positions(self, joint_positions):
        """
        Resets the robot's arm joints to the specified positions.
        """
        for i, joint_index in enumerate(self.arm_joint_indices):
            p.resetJointState(self.robot_id, joint_index, joint_positions[i])

    def get_ee_position_W(self):
        """
        Returns the end-effector position in world coordinates.
        This should use the IK target link for consistency.
        """
        ls = p.getLinkState(self.robot_id, self.ee_link_index, computeForwardKinematics=True)
        return np.array(ls[4])

    def get_joint_positions(self):
        """Returns the current positions of the arm joints."""
        return np.array([p.getJointState(self.robot_id, j)[0] for j in self.arm_joint_indices])

    def get_joint_velocities(self):
        """Returns the current velocities of the arm joints."""
        return np.array([p.getJointState(self.robot_id, j)[1] for j in self.arm_joint_indices])

    def calculate_ik(self, target_pos_W):
        target_pos_W = np.asarray(target_pos_W, dtype=float).tolist()
        self._last_ik_target_pos = np.array(target_pos_W, dtype=float)
        target_orn = self._ik_target_quat if self.ik_use_orientation else None

        if target_orn is None:
            sol = p.calculateInverseKinematics(
                self.robot_id,
                self.ee_link_index,
                target_pos_W,
                lowerLimits=self.joint_limits_lower,
                upperLimits=self.joint_limits_upper,
                jointRanges=self.joint_ranges,
                restPoses=self.joint_rest_poses,
                maxNumIterations=self.ik_max_iters,
                residualThreshold=self.ik_residual_threshold,
            )
        else:
            sol = p.calculateInverseKinematics(
                self.robot_id,
                self.ee_link_index,
                target_pos_W,
                target_orn,
                lowerLimits=self.joint_limits_lower,
                upperLimits=self.joint_limits_upper,
                jointRanges=self.joint_ranges,
                restPoses=self.joint_rest_poses,
                maxNumIterations=self.ik_max_iters,
                residualThreshold=self.ik_residual_threshold,
            )
        return np.array(sol[: len(self.arm_joint_indices)], dtype=float)

    def apply_joint_positions(self, joint_positions):
        # Per-joint POSITION_CONTROL with explicit gains/forces
        for i, j_idx in enumerate(self.arm_joint_indices[: len(joint_positions)]):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=j_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=float(joint_positions[i]),
                force=float(self.joint_max_forces[i] * self.arm_max_force_scale),
                positionGain=self.arm_pos_gain,
                velocityGain=self.arm_vel_gain,
            )
        # Debug EE error vs last IK target (pos + orientation)
        ee = p.getLinkState(self.robot_id, self.ee_link_index, computeForwardKinematics=True)
        ee_pos = np.array(ee[4], dtype=float)
        ee_orn = np.array(ee[5], dtype=float)
        pos_err = float(np.linalg.norm(ee_pos - getattr(self, "_last_ik_target_pos", ee_pos)))
        orn_err = None
        if self.ik_use_orientation and hasattr(self, "_ik_target_quat"):
            # quaternion angle error (deg)
            dot = float(abs(np.dot(ee_orn, np.array(self._ik_target_quat))))
            dot = max(min(dot, 1.0), 0.0)
            orn_err = float(2.0 * np.arccos(dot) * 180.0 / np.pi)

    def force_fingers_closed(self):
        """Applies strong force to ensure fingers remain closed."""
        for j in self.finger_joint_indices:
            p.setJointMotorControl2(
                self.robot_id,
                j,
                p.POSITION_CONTROL,
                targetPosition=self.finger_closed_pos,
                force=self.finger_max_force,
                positionGain=self.finger_kp,
                maxVelocity=self.finger_max_vel,
            )
            # Safety check to snap back if drifted
            if abs(p.getJointState(self.robot_id, j)[0] - self.finger_closed_pos) > 1e-5:
                p.resetJointState(self.robot_id, j, self.finger_closed_pos, 0.0)

    def randomize_dynamics(self, linear_damp: float, angular_damp: float, lateral_friction: float):
        """
        Apply per-link dynamics for the arm and gripper. Values are scalars.
        """
        for link_id in self.arm_joint_indices + [self.ee_link_index]:
            if link_id is not None and link_id >= 0:
                p.changeDynamics(
                    self.robot_id,
                    link_id,
                    linearDamping=float(linear_damp),
                    angularDamping=float(angular_damp),
                    lateralFriction=float(lateral_friction),
                )
        # expose for observations (fail loudly if missing)
        self.joint_damping = [float(linear_damp)] * len(self.arm_joint_indices)
        self.joint_friction = [float(lateral_friction)] * len(self.arm_joint_indices)

    def set_initial_joint_positions(self, initial_qpos=None):
        initial_qpos = np.array(initial_qpos)
        # Apply it
        for i, j in enumerate(self.arm_joint_indices[: len(initial_qpos)]):
            p.resetJointState(self.robot_id, j, float(initial_qpos[i]), 0.0)

    # Debug (not called)
    # --- Inspect/link pose and orientation error ---
    def get_ee_pose_W(self):
        """Returns end-effector (current control link) pose (pos, quat) in world frame."""
        ls = p.getLinkState(self.robot_id, self.ee_link_index, computeForwardKinematics=True)
        return np.array(ls[0], dtype=float), np.array(ls[1], dtype=float)  # (pos, quat xyzw)

    @staticmethod
    def _normalize_quat_xyzw(q):
        q = np.asarray(q, dtype=float)
        n = np.linalg.norm(q)
        return q if n == 0.0 else (q / n)

    def quat_angle_error_deg(self, q_a, q_b):
        """
        Returns the absolute rotation angle (deg) between two quaternions (xyzw).
        Uses 2*acos(|dot|) with normalization and double-cover handling.
        """
        a = self._normalize_quat_xyzw(q_a)
        b = self._normalize_quat_xyzw(q_b)
        dot = float(np.clip(np.abs(np.dot(a, b)), -1.0, 1.0))
        return float(2.0 * np.degrees(np.arccos(dot)))
