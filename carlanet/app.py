import carla
import time
import cv2
import numpy as np
import functools
from .simulator.SimulatorManager import SimulatorManager
from .simulator.Vehicle import Vehicle
from .simulator.sensors.Camera import Camera
from .simulator.sensors.DepthCamera import DepthCamera
from .simulator.sensors.Lidar import Lidar
from .simulator.VehicleController import VehicleController

def handle_keyboard_interrupt(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            func(self, *args, **kwargs)
            self.cleanup()
        except KeyboardInterrupt:
            print("KeyboardInterrupt detected. Exiting gracefully...")
            self.cleanup()  # Call the cleanup method of the instance
    return wrapper

class Application:
    """
    This class manages the main application flow of a CARLA simulation. It initializes a connection to the 
    CARLA simulator, spawns a vehicle, attaches a camera sensor to the vehicle, runs the simulation for a 
    specified duration, and finally cleans up the simulation.

    Attributes:
        simulator_manager (SimulatorManager): An instance of SimulatorManager which handles the 
        interactions with the CARLA simulator.
    """
    def __init__(self, host='localhost', port=2000):
        self.sm = SimulatorManager(host, port)
        self.sm.set_sync_mode(True)
    
    @handle_keyboard_interrupt
    def run(self):
        """
        Runs the main application flow.
        """
        pass

    @handle_keyboard_interrupt
    def test_run(self):
        """
        Runs the test application flow.
        """
        # Spawn a vehicle
        vehicle = Vehicle(self.sm, vehicle_type='model3')

        # Create vehicle controller with your chosen parameters
        vehicle_controller = VehicleController(Kp_longitudinal=1.0, Ki_longitudinal=0.0, Kd_longitudinal=0.0, Kp_lateral=0.00001)

        # Define desired speed in m/s
        desired_speed = 15.0  # 10 m/s

        # Generate waypoints
        waypoints = self.sm.generate_example_waypoints(vehicle.get_actor().get_transform().location, 5, 200)

        # Start at the first waypoint
        next_waypoint = waypoints[0]

        # Spawn a camera sensor and attach it to the vehicle
        # def camera_callback(image):
        #     # Convert image to OpenCV format
        #     frame = Camera.process_data(image)
        #     # print(frame.shape)
        #     cv2.imshow("", frame)
        #     cv2.waitKey(1)
        #     print(f"Camera captured an image at timestamp {image.timestamp}")

        # vehicle.attach_sensor(Camera, callback=camera_callback, transform=carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15)))

        # Move the spectator to the vehicle
        self.sm.move_spectator(vehicle.get_actor())

        # Run the simulation for a certain duration
        simulation_duration = 60  # seconds
        start_time = time.time()
        while time.time() - start_time < simulation_duration:
            time.sleep(0.05)  # Sleep for 0.05 seconds (i.e., 20Hz update rate)

            debug = self.sm.get_world().debug

            for i in range(len(waypoints) - 1):
                # Draw a line between each pair of waypoints
                debug.draw_line(waypoints[i].transform.location, waypoints[i + 1].transform.location, 
                                thickness=0.1, color=carla.Color(255, 0, 0), life_time=0.1, persistent_lines=False)

                # Draw a string with the waypoint number at each waypoint location
                debug.draw_string(waypoints[i].transform.location, f'WP {i}', 
                                draw_shadow=False, color=carla.Color(0, 0, 255), life_time=0.1, persistent_lines=False) 

            # Apply control to the vehicle (e.g., throttle, steering, etc.)
            current_transform = vehicle.get_actor().get_transform()
            current_speed = vehicle.get_velocity()
            current_yaw = current_transform.rotation.yaw
            current_position = current_transform.location

            # Get the control commands
            vehicle_control = vehicle_controller.control(
                desired_speed,
                current_speed,
                current_position,
                current_yaw,
                next_waypoint,
                dt=0.05  # Assuming a control rate of 20Hz (i.e., a control command every 0.05 seconds)
            )

            # Apply control to the vehicle
            vehicle.apply_control(vehicle_control=vehicle_control)

            # Move on to the next waypoint if we're close enough to the current one
            if vehicle.get_actor().get_transform().location.distance(next_waypoint.transform.location) < 1.0:
                if len(waypoints) > 0:
                    next_waypoint = waypoints.pop(0)
            
            print(f"Applied control: {vehicle_control}")
            print(f"Current speed: {current_speed}")

            self.sm.move_spectator(vehicle.get_actor())
            
            # Tick the CARLA simulator
            self.sm.get_world().tick()
    
    def cleanup(self):
        """
        Cleans up the simulation.
        """
        self.sm.destroy()
