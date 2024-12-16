import numpy as np


class PIDController:
    def __init__(self, dim=3, dt=0.05):
        # controller = PIDController(dim=3, dt=0.05)
        # Initialize gains (tuned for position control in mm)
        """
        alexs stuff
        self.Kp = 0.5 * np.eye(dim)  # Proportional gain
        self.Ki = 0.05 * np.eye(dim) # Integral gain
        self.Kd = 0.1 * np.eye(dim)  # Derivative gain
        """
        # Kd=0.1kp
        # ki=0.01Kp
        self.Kp = 0.5 * np.eye(dim)  # Proportional gain
        self.Ki = 0.05 * np.eye(dim)  # Integral gain
        self.Kd = 0.1 * np.eye(dim)  # Derivative gain

        # Initialize error terms

        self.error_integral = np.zeros(dim)
        self.error_prev = np.zeros(dim)
        self.dt = dt  # Control period in seconds

    def compute_pid(self, error):
        # Proportional term
        p_term = self.Kp @ error

        # Integral term (accumulating errors over time)
        self.error_integral += error * self.dt
        i_term = self.Ki @ self.error_integral

        # Derivative term (rate of change of error)
        d_term = self.Kd @ ((error - self.error_prev) / self.dt)

        # PID output
        output = p_term + i_term + d_term

        # Update previous error
        self.error_prev = error

        return output

    def reset(self):
        """Resets the integral and previous error terms to zero."""
        self.error_integral = 0.0
        self.error_prev = 0.0

    def update(self, error):
        """
        Computes the PID control output for a given error.

        Parameters:
            error (float): The current error.

        Returns:
            float: The control output for this error.
        """
        # Proportional term
        p_term = self.Kp * error

        # Integral term (accumulating errors over time)
        self.error_integral += error * self.dt
        i_term = self.Ki * self.error_integral

        # Derivative term (rate of change of error)
        d_term = self.Kd * ((error - self.error_prev) / self.dt)

        # PID output
        output = p_term + i_term + d_term

        # Update previous error
        self.error_prev = error

        return output
