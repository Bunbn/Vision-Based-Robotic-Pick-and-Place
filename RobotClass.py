import numpy as np
import math
from classes.OM_X_arm import OM_X_arm
from classes.DX_XM430_W350 import DX_XM430_W350


class Robot(OM_X_arm):
    """Robot class for controlling the OpenManipulator-X Robot.
    Inherits from OM_X_arm and provides methods specific to the robot's operation.
    """

    def __init__(self, online_mode=True):
        """Initialize the Robot class.
        Creates constants and connects via serial. Sets default mode and state.

        Args:
            online_mode (bool, optional): Whether to initialize in online mode. Defaults to True.
        """
        if online_mode:
            super().__init__()

            # Set default mode and state
            # Change robot to position mode with torque enabled by default
            self.write_mode("position")
            self.write_motor_state(True)

            # Set the robot to move between positions with a 5 second trajectory profile
            self.write_time(5)

        # Robot dimensions (in mm)
        self.mDim = [77, 130, 124, 126]
        self.mOtherDim = [128, 24]

        # Robot joint limits (in degrees)
        self.joint_limits = [
            [-180, 180],
            [-115, 90],
            [-90, 88],
            [-100, 115]
        ]

        # DH parameter table
        self.DH = np.array([
            [0, self.mDim[0], 0, -np.pi/2],
            [-np.atan2(self.mOtherDim[0], self.mOtherDim[1]),
             0, self.mDim[1], 0],
            [np.atan2(self.mOtherDim[0], self.mOtherDim[1]),
             0, self.mDim[2], 0],
            [0, 0, self.mDim[3], 0]
        ])

    def write_joints(self, goals):
        """Send joints to the desired angles.

        Args:
            goals (list): List of angles (degrees) for each joint.
        """
        goals = [
            round(goal * DX_XM430_W350.TICKS_PER_DEG +
                  DX_XM430_W350.TICK_POS_OFFSET)
            % DX_XM430_W350.TICKS_PER_ROT
            for goal in goals
        ]
        self.bulk_read_write(DX_XM430_W350.POS_LEN,
                             DX_XM430_W350.GOAL_POSITION, goals)

    def write_time(self, time, acc_time=None):
        """Create a time-based profile (trapezoidal) based on desired times.

        Args:
            time (float): Total profile time in seconds. If 0, profile will be disabled.
            acc_time (float, optional): Total acceleration time for ramp up/down. Defaults to time/3.
        """
        if acc_time is None:
            acc_time = time / 3

        time_ms = int(time * DX_XM430_W350.MS_PER_S)
        acc_time_ms = int(acc_time * DX_XM430_W350.MS_PER_S)

        self.bulk_read_write(
            DX_XM430_W350.PROF_ACC_LEN,
            DX_XM430_W350.PROF_ACC,
            [acc_time_ms] * self.motorsNum
        )
        self.bulk_read_write(
            DX_XM430_W350.PROF_VEL_LEN,
            DX_XM430_W350.PROF_VEL,
            [time_ms] * self.motorsNum
        )

    def write_gripper(self, open):
        """Set the gripper to open or closed position.

        Args:
            open (bool): True to open gripper, False to close.
        """
        position = -45 if open else 45
        self.gripper.write_position(position)

    def read_gripper(self):
        """Read current gripper position.

        Returns:
            float: Current gripper position.
        """
        return self.gripper.read_position()

    def write_motor_state(self, enable):
        """Set position holding for joints on/off.

        Args:
            enable (bool): True to enable torque, False to disable.
        """
        state = 1 if enable else 0
        states = [state] * self.motorsNum
        self.bulk_read_write(
            DX_XM430_W350.TORQUE_ENABLE_LEN,
            DX_XM430_W350.TORQUE_ENABLE,
            states
        )

    def write_currents(self, currents):
        """Supply joints with desired currents.

        Args:
            currents (list): List of currents (mA) for each joint.
        """
        current_in_ticks = [
            round(current * DX_XM430_W350.TICKS_PER_mA)
            for current in currents
        ]
        self.bulk_read_write(
            DX_XM430_W350.CURR_LEN,
            DX_XM430_W350.GOAL_CURRENT,
            current_in_ticks
        )

    def write_mode(self, mode):
        """Change operating mode for all joints.

        Args:
            mode (str): New operating mode. Options: "current", "velocity", "position",
                       "ext position", "curr position", "pwm voltage"

        Raises:
            ValueError: If mode is invalid.
        """
        mode_map = {
            "current": DX_XM430_W350.CURR_CNTR_MD,
            "c": DX_XM430_W350.CURR_CNTR_MD,
            "velocity": DX_XM430_W350.VEL_CNTR_MD,
            "v": DX_XM430_W350.VEL_CNTR_MD,
            "position": DX_XM430_W350.POS_CNTR_MD,
            "p": DX_XM430_W350.POS_CNTR_MD,
            "ext position": DX_XM430_W350.EXT_POS_CNTR_MD,
            "ep": DX_XM430_W350.EXT_POS_CNTR_MD,
            "curr position": DX_XM430_W350.CURR_POS_CNTR_MD,
            "cp": DX_XM430_W350.CURR_POS_CNTR_MD,
            "pwm voltage": DX_XM430_W350.PWM_CNTR_MD,
            "pwm": DX_XM430_W350.PWM_CNTR_MD
        }

        if mode not in mode_map:
            raise ValueError(
                f"writeMode input cannot be '{mode}'. See implementation in DX_XM430_W350 class.")

        write_mode = mode_map[mode]
        self.write_motor_state(False)
        write_modes = [write_mode] * self.motorsNum
        self.bulk_read_write(
            DX_XM430_W350.OPR_MODE_LEN,
            DX_XM430_W350.OPR_MODE,
            write_modes
        )
        self.write_motor_state(True)

    def get_joints_readings(self):
        """Get current joint positions, velocities, and currents.

        Returns:
            numpy.ndarray: 3x4 array of positions (deg), velocities (deg/s), and currents (mA).
        """
        readings = np.zeros((3, 4))

        positions = np.array(
            self.bulk_read_write(DX_XM430_W350.POS_LEN,
                                 DX_XM430_W350.CURR_POSITION)
        )
        velocities = np.array(
            self.bulk_read_write(DX_XM430_W350.VEL_LEN,
                                 DX_XM430_W350.CURR_VELOCITY)
        )
        currents = np.array(
            self.bulk_read_write(DX_XM430_W350.CURR_LEN,
                                 DX_XM430_W350.CURR_CURRENT)
        )

        # Handle two's complement
        velocities[velocities > 0x7FFFFFFF] -= 4294967296
        currents[currents > 0x7FFF] -= 65536

        readings[0, :] = (
            positions - DX_XM430_W350.TICK_POS_OFFSET) / DX_XM430_W350.TICKS_PER_DEG
        readings[1, :] = velocities / DX_XM430_W350.TICKS_PER_ANGVEL
        readings[2, :] = currents / DX_XM430_W350.TICKS_PER_mA

        return readings

    def write_velocities(self, vels):
        """Send joints to desired velocities.

        Args:
            vels (list): List of angular velocities (deg/s) for each joint.
        """
        vels = [round(vel * DX_XM430_W350.TICKS_PER_ANGVEL) for vel in vels]
        self.bulk_read_write(DX_XM430_W350.VEL_LEN,
                             DX_XM430_W350.GOAL_VELOCITY, vels)

    def get_dh_row_mat(self, one_joint_dh_params):
        """Calculate intermediate transformation A_i for a DH parameter table row.

        Args:
            one_joint_dh_params (list): DH parameters [theta, d, a, alpha] for one joint.

        Returns:
            numpy.ndarray: 4x4 homogeneous transformation matrix.
        """
        theta, d, a, alpha = one_joint_dh_params

        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)

        return np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0,       sa,       ca,      d],
            [0,        0,        0,      1]
        ])

    def get_int_mat(self, joint_angles):
        """Calculate intermediate matrices for given joint angles.

        Args:
            joint_angles (list): List of joint angles in degrees.

        Returns:
            numpy.ndarray: 4x4x4 array of intermediate transformation matrices.
        """
        joint_angles = np.deg2rad(joint_angles)
        a_matrices = np.zeros((4, 4, 4))

        offset_angle = np.pi/2 - np.arcsin(self.mOtherDim[1]/self.mDim[1])

        dh_params = [
            [joint_angles[0], self.mDim[0], 0, -np.pi/2],
            [joint_angles[1] - offset_angle, 0, self.mDim[1], 0],
            [joint_angles[2] + offset_angle, 0, self.mDim[2], 0],
            [joint_angles[3], 0, self.mDim[3], 0]
        ]

        for i in range(4):
            a_matrices[i] = self.get_dh_row_mat(dh_params[i])

        return a_matrices

    def get_acc_mat(self, joint_angles):
        """Calculate accumulative transformations for all joints.

        Args:
            joint_angles (list): List of joint angles in degrees.

        Returns:
            numpy.ndarray: 4x4x4 array of transformation matrices.
        """
        a_i = self.get_int_mat(joint_angles)

        t_matrices = np.zeros((4, 4, 4))
        t_matrices[0] = a_i[0]

        for i in range(1, 4):
            t_matrices[i] = t_matrices[i-1] @ a_i[i]

        return t_matrices

    def get_fk(self, joint_angles):
        """Calculate forward kinematics for given joint angles.

        Args:
            joint_angles (list): List of joint angles in degrees.

        Returns:
            numpy.ndarray: 4x4 homogeneous transformation matrix.
        """
        return self.get_acc_mat(joint_angles)[3]

    def get_current_fk(self):
        """Get current end-effector transformation based on current joint angles.

        Returns:
            numpy.ndarray: 4x4 homogeneous transformation matrix.
        """
        return self.get_fk(self.get_joints_readings()[0])

    def get_ee_pos(self, joint_angles):
        """Calculate end-effector position and orientation for given joint angles.

        Args:
            joint_angles (list): List of joint angles in degrees.

        Returns:
            numpy.ndarray: Array [x, y, z, pitch, yaw] of position (mm) and orientation (deg).
        """
        t_ee_base = self.get_fk(joint_angles)

        position = t_ee_base[:3, 3]
        pitch = -np.sum(joint_angles[1:])
        yaw = joint_angles[0]

        return np.array([*position, pitch, yaw])

    def get_ik(self, ee_pos, prefer_elbow_up=True):
        if len(ee_pos) == 3:
            x, y, z = ee_pos
            pitch = 0  # Default value if pitch is not provided
        elif len(ee_pos) == 4:
            x, y, z, pitch = ee_pos
        else:
            raise ValueError(
                "End-effector position must have 3 or 4 elements.")
        # Convert pitch to radians
        alpha = np.deg2rad(pitch)

        # Unpack link lengths
        L1, L2, L3, L4 = self.mDim

        # 1st calculate r, rw, and zw
        r = np.sqrt(x**2 + y**2)
        rw = r - (L4 * np.cos(alpha))
        zw = z - L1 - (L4 * np.sin(alpha))

        # 2nd calculate dw
        dw = np.sqrt(rw**2 + zw**2)
        # Handle error if desired end effector pose is unreachable
        max_reach = L1 + L2+L3+L4  # Example calculation for a 2-link arm
        # Euclidean distance from the origin
        distance = (x**2 + y**2 + z**2) ** 0.5

        if distance > max_reach:
            print(distance)
            # raise ValueError("Desired end effector pose unreachable (exceeds maximum arm reach)")

        # 3rd calculate intermediate angles
        mu = np.arctan2(zw, rw)
        beta = np.arccos((L2**2 + L3**2 - dw**2) / (2 * L2 * L3))
        beta2 = -beta
        gamma = np.arccos((dw**2 + L2**2 - L3**2) / (2 * dw * L2))
        gamma2 = -gamma
        delta = np.arctan2(self.mOtherDim[1], self.mOtherDim[0])

        # 4th calculate joint angles
        # 4.0 q1 doesn't depend on elbow-up or elbow-down
        q1 = np.arctan2(y, x)
        # 4.1 get elbow-up solution
        q2_up = np.pi / 2 - delta - gamma - mu
        q3_up = np.pi / 2 + delta - beta
        q4_up = -alpha - q2_up - q3_up
        # 4.2 get elbow-down solution
        q2_down = np.pi / 2 - delta - gamma2 - mu
        q3_down = np.pi / 2 + delta - beta2
        q4_down = -alpha - q2_down - q3_down

        # Convert to degrees
        q1_deg, q2_up_deg, q3_up_deg, q4_up_deg = np.rad2deg(
            [q1, q2_up, q3_up, q4_up])
        q1_deg, q2_down_deg, q3_down_deg, q4_down_deg = np.rad2deg(
            [q1, q2_down, q3_down, q4_down]
        )
        # Return elbow-up when possible
        # if prefer_elbow_up and elbow_up_valid:
        return np.array([q1_deg, q2_up_deg, q3_up_deg, q4_up_deg])
        # Return elbow-down if elbow-up invalid regardless of preference

    def get_jacobian(self, q):
        acc_matrices = self.get_acc_mat(q)

        # z = [acc_matrices[i][:3, 2] for i in range(len(acc_matrices))]
        # o = [acc_matrices[i][:3, 3] for i in range(len(acc_matrices))]
        J = np.zeros((6, 4))
        z = [np.array([0, 0, 1])]  # Base z-axis
        o = [np.array([0, 0, 0])]  # Base origin

        # Append all other frames' z-axes and positions
        o += [acc_matrices[i][:3, 3] for i in range(len(acc_matrices))]
        z += [acc_matrices[i][:3, 2] for i in range(len(acc_matrices))]
        J = np.zeros((6, 4))  # 4 joints + base

        # Calculate each column of the Jacobian
        for i in range(4):  # Now iterates over the base and 4 joints
            J[:3, i] = (np.cross(z[i], o[4] - o[i])) * \
                (np.pi/180)  # Linear velocity component
            J[3:, i] = z[i]  # Angular velocity component (rotation axis)
            # if i==2 or i==1:
            # J[i,:]=np.rad2deg(J[i,:])*-10
        # print(J[1][0])

        return J

    def get_ik(self, ee_pos, prefer_elbow_up=True):
        """
        Calculate inverse kinematics for a robotic arm.

        Args:
            ee_pos (list): End-effector position, either [x, y, z] or [x, y, z, pitch]
            prefer_elbow_up (bool): Whether to prefer elbow-up solution if possible

        Returns:
            numpy.ndarray: Joint angles in degrees [q1, q2, q3, q4]

        Raises:
            ValueError: If ee_pos doesn't have 3 or 4 elements or if position is unreachable
        """
        # Parse input position and pitch
        if len(ee_pos) == 3:
            x, y, z = ee_pos
            pitch = 0  # Default value if pitch is not provided
        elif len(ee_pos) == 4:
            x, y, z, pitch = ee_pos
        else:
            raise ValueError(
                "End-effector position must have 3 or 4 elements.")

        # Convert pitch to radians
        alpha = np.deg2rad(pitch)

        # Unpack link lengths
        L1, L2, L3, L4 = self.mDim

        # Calculate intermediate positions
        r = np.sqrt(x**2 + y**2)
        rw = r - (L4 * np.cos(alpha))
        zw = z - L1 - (L4 * np.sin(alpha))

        # Calculate wrist position distance
        dw = np.sqrt(rw**2 + zw**2)

        # Check reachability
        max_reach = L1 + L2 + L3 + L4
        distance = (x**2 + y**2 + z**2) ** 0.5

        if distance > max_reach:
            print(distance)
            # raise ValueError("Desired end effector pose unreachable (exceeds maximum arm reach)")

        # Calculate intermediate angles
        mu = np.arctan2(zw, rw)
        beta = np.arccos((L2**2 + L3**2 - dw**2) / (2 * L2 * L3))
        beta2 = -beta
        gamma = np.arccos((dw**2 + L2**2 - L3**2) / (2 * dw * L2))
        gamma2 = -gamma
        delta = np.arctan2(self.mOtherDim[1], self.mOtherDim[0])

        # Calculate joint angles
        # Base rotation
        q1 = np.arctan2(y, x)

        # Elbow-up solution
        q2_up = np.pi / 2 - delta - gamma - mu
        q3_up = np.pi / 2 + delta - beta
        q4_up = -alpha - q2_up - q3_up

        # Elbow-down solution
        q2_down = np.pi / 2 - delta - gamma2 - mu
        q3_down = np.pi / 2 + delta - beta2
        q4_down = -alpha - q2_down - q3_down

        # Convert all angles to degrees
        q1_deg, q2_up_deg, q3_up_deg, q4_up_deg = np.rad2deg(
            [q1, q2_up, q3_up, q4_up])
        q1_deg, q2_down_deg, q3_down_deg, q4_down_deg = np.rad2deg(
            [q1, q2_down, q3_down, q4_down])

        # Return preferred solution (currently always returns elbow-up)
        return np.array([q1_deg, q2_up_deg, q3_up_deg, q4_up_deg])

    def get_jacobian(self, q):
        """
        Calculate the Jacobian matrix for the robot arm.

        Args:
            q (numpy.ndarray): Joint angles

        Returns:
            numpy.ndarray: 6x4 Jacobian matrix containing linear and angular velocity components
        """
        acc_matrices = self.get_acc_mat(q)

        # Initialize base frame
        z = [np.array([0, 0, 1])]  # Base z-axis
        o = [np.array([0, 0, 0])]  # Base origin

        # Get z-axes and origins for all frames
        o += [acc_matrices[i][:3, 3] for i in range(len(acc_matrices))]
        z += [acc_matrices[i][:3, 2] for i in range(len(acc_matrices))]

        # Initialize Jacobian matrix
        J = np.zeros((6, 4))

        # Calculate Jacobian columns
        for i in range(4):
            # Linear velocity component (cross product of rotation axis and distance to end-effector)
            J[:3, i] = (np.cross(z[i], o[4] - o[i])) * (np.pi/180)
            # Angular velocity component (rotation axis)
            J[3:, i] = z[i]

        return J
