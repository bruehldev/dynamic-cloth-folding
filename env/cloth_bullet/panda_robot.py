import pybullet as p
import numpy as np
import os

class PandaRobot:
    """
    Encapsulates the Franka Emika Panda robot logic, including loading,
    joint/link identification, control, and sensor feedback.
    """
    def __init__(self, base_position, base_orientation):
        self.urdf_path = "franka_panda/panda.urdf"
        self.robot_id = p.loadURDF(self.urdf_path, base_position, base_orientation, useFixedBase=True)

        self._find_links_and_joints()
        self._get_joint_limits()
        self._setup_finger_control()
        self.set_initial_joint_positions()
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
            link_name = info[12].decode('UTF-8')
            joint_name = info[1].decode('UTF-8')

            if info[2] == p.JOINT_REVOLUTE:
                self.arm_joint_indices.append(j)
            elif info[2] == p.JOINT_PRISMATIC and 'finger' in joint_name:
                self.finger_joint_indices.append(j)
                p.setCollisionFilterGroupMask(self.robot_id, j, 1, 0)

            if link_name == 'panda_leftfinger': self.left_finger_link_index = j
            elif link_name == 'panda_rightfinger': self.right_finger_link_index = j
            elif link_name == 'panda_hand': self.hand_link_index = j
            elif link_name == 'panda_grasptarget': self.grasp_target_link_index = j

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

    def _setup_finger_control(self):
        """
        Sets up motor control for the robot's fingers and welds them shut.
        """
        self.finger_closed_pos = 0.0
        self.finger_max_force = 200.0 # Default strong force
        self.finger_kp = 1.0 # Default position gain
        self.finger_max_vel = 2.0 # Default max velocity

    def weld_fingers_shut(self):
        """Creates fixed constraints to weld the fingers to the hand, ensuring a rigid grip."""
        if self.hand_link_index is not None:
            if self.left_finger_link_index is not None:
                cid = p.createConstraint(self.robot_id, self.hand_link_index,
                                   self.robot_id, self.left_finger_link_index,
                                   jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
                                   parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
                p.changeConstraint(cid, maxForce=self.finger_max_force)
            if self.right_finger_link_index is not None:
                cid = p.createConstraint(self.robot_id, self.hand_link_index,
                                   self.robot_id, self.right_finger_link_index,
                                   jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
                                   parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
                p.changeConstraint(cid, maxForce=self.finger_max_force)

    def reset_to_neutral(self):
        """
        Resets the robot's arm joints to a neutral pose (all zeros).
        """
        for j in self.arm_joint_indices:
            p.resetJointState(self.robot_id, j, 0.0, 0.0)
            p.changeDynamics(self.robot_id, j, linearDamping=0.1, angularDamping=0.1)
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

    def calculate_ik(self, target_pos):
        """
        Calculates the joint positions needed to reach a target end-effector position.
        This must use the ee_link_index (grasptarget) to move the point between
        the fingers to the target.
        """
        return p.calculateInverseKinematics(
            self.robot_id,
            self.ee_link_index,
            target_pos,
            lowerLimits=self.joint_limits_lower,
            upperLimits=self.joint_limits_upper,
            jointRanges=self.joint_ranges,
            restPoses=self.joint_rest_poses,
            maxNumIterations=100,
            residualThreshold=1e-5,
        )

    def apply_joint_positions(self, joint_positions):
        """
        Applies target joint positions to the robot's arm controllers.
        """
        p.setJointMotorControlArray(
            self.robot_id, self.arm_joint_indices[:7], p.POSITION_CONTROL,
            targetPositions=joint_positions[:7], forces=self.joint_max_forces[:7]
        )

    def force_fingers_closed(self):
        """Applies strong force to ensure fingers remain closed."""
        for j in self.finger_joint_indices:
            p.setJointMotorControl2(
                self.robot_id, j, p.POSITION_CONTROL,
                targetPosition=self.finger_closed_pos, force=self.finger_max_force,
                positionGain=self.finger_kp, maxVelocity=self.finger_max_vel
            )
            # Safety check to snap back if drifted
            if abs(p.getJointState(self.robot_id, j)[0] - self.finger_closed_pos) > 1e-5:
                p.resetJointState(self.robot_id, j, self.finger_closed_pos, 0.0)

    def randomize_dynamics(self, linear_damp: float, angular_damp: float, lateral_friction: float):
        """
        Apply simple per-link damping and contact friction to approximate MuJoCo
        dynamics randomization for the manipulator.
        """
        # Arm links
        for j in self.arm_joint_indices:
            p.changeDynamics(self.robot_id, j,
                             linearDamping=float(linear_damp),
                             angularDamping=float(angular_damp),
                             lateralFriction=float(lateral_friction))
        # Hand/EE (if present)
        for maybe in [self.hand_link_index, self.ee_link_index]:
            if maybe is not None and maybe >= 0:
                p.changeDynamics(self.robot_id, int(maybe),
                                 linearDamping=float(linear_damp),
                                 angularDamping=float(angular_damp),
                                 lateralFriction=float(lateral_friction))

    def set_initial_joint_positions(self):
        initial_qpos = np.array(
            [0.212422, 0.362907, -0.00733391, -1.9649, -0.0198034, 2.37451, -1.50499])
        # Apply it
        for i, j in enumerate(self.arm_joint_indices[:len(initial_qpos)]):
            p.resetJointState(self.robot_id, j, float(initial_qpos[i]), 0.0)