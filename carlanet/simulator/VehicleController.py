import carla
import numpy as np
from simple_pid import PID

class VehicleController:
    def __init__(self, Kp_longitudinal, Ki_longitudinal, Kd_longitudinal, Kp_lateral):
        """
        Initializes the vehicle controller.

        Parameters:
            Kp_longitudinal (float): The proportional gain for the longitudinal PID controller.
            Ki_longitudinal (float): The integral gain for the longitudinal PID controller.
            Kd_longitudinal (float): The derivative gain for the longitudinal PID controller.
            Kp_lateral (float): The proportional gain for the lateral (Stanley) controller.
        """
        self.pid_longitudinal = PID(Kp_longitudinal, Ki_longitudinal, Kd_longitudinal)
        self.Kp_lateral = Kp_lateral

    def longitudinal_control(self, desired_speed, current_speed, dt):
        """
        Compute the longitudinal control command (throttle/brake) for the vehicle.

        Parameters:
            desired_speed (float): The desired speed for the vehicle.
            current_speed (float): The current speed of the vehicle.
            dt (float): The time step.

        Returns:
            float: The throttle/brake command for the vehicle.
        """
        self.pid_longitudinal.setpoint = desired_speed
        throttle_brake = self.pid_longitudinal(current_speed, dt)
        return carla.VehicleControl(throttle=throttle_brake if throttle_brake >= 0 else 0.0, 
                                    brake=-throttle_brake if throttle_brake < 0 else 0.0)

    def lateral_control(self, current_position: carla.Location, current_yaw: float, next_waypoint: carla.Waypoint, current_speed: float):
        """
        Compute the lateral control command (steering angle) for the vehicle using Stanley controller.

        Parameters:
            current_position (carla.Location): The current position of the vehicle.
            current_yaw (float): The current yaw angle of the vehicle (orientation with respect to the x-axis).
            next_waypoint (carla.Waypoint): The next waypoint the vehicle should aim for.

        Returns:
            carla.VehicleControl: The steering angle command for the vehicle.
        """
        k_s = 2.5

        # Calculate the target yaw based on the direction to the next waypoint
        target_yaw = np.arctan2(next_waypoint.transform.location.y - current_position.y, 
                                next_waypoint.transform.location.x - current_position.x)

        # Calculate the cross track error
        cross_track_error = np.linalg.norm(np.array([current_position.x - next_waypoint.transform.location.x, 
                                                     current_position.y - next_waypoint.transform.location.y]))

        # Implement the Stanley controller
        yaw_error = target_yaw - np.radians(current_yaw)
        yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))  # ensure the error is -pi to pi

        # Calculate the steering angle
        steering_angle = yaw_error + np.arctan2(self.Kp_lateral * cross_track_error, k_s + current_speed)

        return carla.VehicleControl(steer=np.clip(steering_angle, -1.0, 1.0))

    def control(self, desired_speed, current_speed, current_position, current_yaw, next_waypoint, dt):
        """
        Compute the control commands for the vehicle.

        Parameters:
            desired_speed (float): The desired speed for the vehicle.
            current_speed (float): The current speed of the vehicle.
            current_position (carla.Location): The current position of the vehicle.
            current_yaw (float): The current yaw angle of the vehicle (orientation with respect to the x-axis).
            next_waypoint (carla.Waypoint): The next waypoint the vehicle should aim for.
            dt (float): The time step.

        Returns:
            carla.VehicleControl: The control command for the vehicle.
        """
        throttle_brake_control = self.longitudinal_control(desired_speed, current_speed, dt)
        steering_angle_control = self.lateral_control(current_position, current_yaw, next_waypoint, current_speed)
        
        # Combine throttle, brake, and steering controls
        control = carla.VehicleControl(throttle=throttle_brake_control.throttle, 
                                       brake=throttle_brake_control.brake, 
                                       steer=steering_angle_control.steer)
        return control
