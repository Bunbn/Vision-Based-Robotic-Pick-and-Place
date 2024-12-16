import numpy as np

class TrajPlanner:
    """
    Trajectory Planner class for calculating trajectories for different polynomial orders and relevant coefficients.
    """

    def __init__(self, setpoints):
        """
        Initialize the TrajPlanner class.

        Parameters:
        setpoints (numpy array): List of setpoints to travel to.
        """
        self.setpoints = setpoints  # n x 4 numpy array of setpoints

    def calc_cubic_coeff(self, t0, tf, q0, qf, v0=0, vf=0):
        """
        Calculate cubic polynomial coefficients for a given setpoint.

        :param t0: Start time
        :param tf: End time
        :param q0: Start position
        :param qf: End position
        :param v0: Start velocity (default is 0)
        :param vf: End velocity (default is 0)
        :return: 1x4 numpy array of cubic coefficients [a0, a1, a2, a3]
        """
        # Setup matrices to solve for coefficients
        A = np.array([
            [1, t0, t0**2,   t0**3],
            [0,  1,  2*t0, 3*t0**2],
            [1, tf, tf**2,   tf**3],
            [0,  1,  2*tf, 3*tf**2]
        ])

        b = np.array([q0, v0, qf, vf])

        # Solve for coefficients a0, a1, a2, a3
        coeff = np.linalg.solve(A, b)

        # Print coefficients for debugging
        print(f"Cubic Coefficients: {coeff}")

        return coeff

    def calc_cubic_traj(self, time_span, n, coeff):
        """
        Calculate the cubic trajectory between two points.

        :param time_span: Time duration between two setpoints
        :param n: Number of intermediate waypoints
        :param coeff: Coefficients of the cubic trajectory (1x4 array)
        :return: (n+2)x1 numpy array of waypoints
        """
        # Create an array of time samples excluding the start point
        t_samples = np.linspace(0, time_span, n + 2)[1:-1]

        # Calculate trajectory for each time sample using the cubic coefficients
        traj = coeff[0] + (coeff[1] * t_samples) + (coeff[2]
                                                    * t_samples**2) + (coeff[3] * t_samples**3)

        # Return as column vector (-1 for automatic number of rows, 1 for exactly 1 column)
        return traj.reshape(-1, 1)

    def get_cubic_traj(self, traj_time, points_num):
        """
        Generate cubic trajectories for all four joints, considering multiple setpoints.

        Parameters:
        traj_time (int): Time between each pair of setpoints.
        points_num (int): Number of intermediate waypoints between setpoints.

        Returns:
        numpy array: A list of waypoints for the cubic trajectory for all joints.
        """
        setpoints = self.setpoints
        num_joints = setpoints.shape[1]  # Number of joints (assumed 4)
        # Number of trajectory segments between setpoints
        num_segments = len(setpoints) - 1

        # Total waypoints is: (num_segments * (points_num + 1)) + 1
        total_waypoints = num_segments * (points_num + 1) + 1
        # Pre-allocate trajectory array
        waypoints_list = np.zeros((total_waypoints, 5))

        # Track the current index in the waypoints_list
        count = 0

        # Loop over all joints
        for joint in range(num_joints):
            count = 0  # Reset count for each joint's trajectory

            # Loop over all segments between setpoints
            for segment in range(num_segments):
                q_start = setpoints[segment, joint]
                q_end = setpoints[segment + 1, joint]

                # Get cubic coefficients for this segment
                coeff = self.calc_cubic_coeff(
                    0, traj_time, q_start, q_end, 0, 0)

                # Add the start point to the waypoints
                # Only add time once for the entire trajectory (first joint)
                if joint == 0:
                    waypoints_list[count, 1:] = setpoints[segment, :]
                count += 1

                # Calculate intermediate waypoints for this segment
                waypoints = self.calc_cubic_traj(traj_time, points_num, coeff)

                # Store the waypoints in the correct position for this joint
                waypoints_list[count:count + points_num,
                               joint + 1] = waypoints.flatten()
                count += points_num

            # Add the final setpoint after the last segment
            waypoints_list[count, 1:] = setpoints[-1, :]

        # Add time to the first column (evenly spaced)
        time = np.linspace(0, traj_time * num_segments, total_waypoints)
        waypoints_list[:, 0] = time

        return waypoints_list

    def calc_quintic_coeff(self, t0, tf, q0, qf, v0=0, vf=0, a0=0, af=0):
        """
        Calculate quintic polynomial coefficients for a given setpoint.

        :param t0: Start time
        :param tf: End time
        :param q0: Start position
        :param qf: End position
        :param v0: Start velocity (default is 0)
        :param vf: End velocity (default is 0)
        :param a0: Start acceleration (default is 0)
        :param af: End acceleration (default is 0)
        :return: 1x6 numpy array of quintic coefficients [a0, a1, a2, a3, a4, a5]
        """
        # Setup matrices to solve for coefficients
        A = np.array([
            [1, t0, t0**2,   t0**3,    t0**4,    t0**5],
            [0,  1,  2*t0, 3*t0**2,  4*t0**3,  5*t0**4],
            [0,  0,     2,    6*t0, 12*t0**2, 20*t0**3],
            [1, tf, tf**2,   tf**3,    tf**4,    tf**5],
            [0,  1,  2*tf, 3*tf**2,  4*tf**3,  5*tf**4],
            [0,  0,     2,    6*tf, 12*tf**2, 20*tf**3]
        ])

        b = np.array([q0, v0, a0, qf, vf, af])

        # Solve for coefficients a0, a1, a2, a3, a4, a5
        coeff = np.linalg.solve(A, b)

        # Print coefficients for debugging
        print(f"Cubic Coefficients: {coeff}")

        return coeff

    def calc_quintic_traj(self, time_span, n, coeff):
        """
        Calculate the quintic trajectory between two points.

        :param time_span: Time duration between two setpoints
        :param n: Number of intermediate waypoints
        :param coeff: Coefficients of the quintic trajectory (1x6 array)
        :return: (n+2)x1 numpy array of waypoints
        """
        # Create an array of time samples excluding the start point
        t_samples = np.linspace(0, time_span, n + 2)[1:-1]

        # Calculate trajectory for each time sample using the cubic coefficients
        traj = coeff[0] + (coeff[1] * t_samples) + (coeff[2] * t_samples**2) + (
            coeff[3] * t_samples**3) + (coeff[4] * t_samples**4) + (coeff[5] * t_samples**5)

        # Return as column vector (-1 for automatic number of rows, 1 for exactly 1 column)
        return traj.reshape(-1, 1)

    def get_quintic_traj(self, traj_time, points_num):
        """
        Generate cubic trajectories for all four joints, considering multiple setpoints.

        Parameters:
        traj_time (int): Time between each pair of setpoints.
        points_num (int): Number of intermediate waypoints between setpoints.

        Returns:
        numpy array: A list of waypoints for the cubic trajectory for all joints.
        """
        setpoints = self.setpoints
        num_joints = setpoints.shape[1]  # Number of joints (assumed 4)
        # Number of trajectory segments between setpoints
        num_segments = len(setpoints) - 1

        # Total waypoints is: (num_segments * (points_num + 1)) + 1
        total_waypoints = num_segments * (points_num + 1) + 1
        # Pre-allocate trajectory array
        waypoints_list = np.zeros((total_waypoints, 5))

        # Track the current index in the waypoints_list
        count = 0

        # Loop over all joints
        for joint in range(num_joints):
            count = 0  # Reset count for each joint's trajectory

            # Loop over all segments between setpoints
            for segment in range(num_segments):
                q_start = setpoints[segment, joint]
                q_end = setpoints[segment + 1, joint]

                # Get cubic coefficients for this segment
                coeff = self.calc_quintic_coeff(
                    0, traj_time, q_start, q_end, 0, 0, 0, 0)

                # Add the start point to the waypoints
                # Only add time once for the entire trajectory (first joint)
                if joint == 0:
                    waypoints_list[count, 1:] = setpoints[segment, :]
                count += 1

                # Calculate intermediate waypoints for this segment
                waypoints = self.calc_quintic_traj(
                    traj_time, points_num, coeff)

                # Store the waypoints in the correct position for this joint
                waypoints_list[count:count + points_num,
                               joint + 1] = waypoints.flatten()
                count += points_num

            # Add the final setpoint after the last segment
            waypoints_list[count, 1:] = setpoints[-1, :]

        # Add time to the first column (evenly spaced)
        time = np.linspace(0, traj_time * num_segments, total_waypoints)
        waypoints_list[:, 0] = time

        return waypoints_list
