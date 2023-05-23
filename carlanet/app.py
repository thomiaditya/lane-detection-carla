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

def handle_keyboard_interrupt(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
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
    
    def run(self):
        """
        Runs the main application flow.
        """
        # Spawn a vehicle
        vehicle = self.sm.spawn_vehicle()

        # Spawn a camera sensor and attach it to the vehicle
        camera_callback = lambda image: print(f"Camera captured an image at timestamp {image.timestamp}")
        self.sm.spawn_sensor('sensor.camera.rgb', camera_callback, vehicle)

        self.handle_cleanup()

    @handle_keyboard_interrupt
    def test_run(self):
        """
        Runs the test application flow.
        """
        # Spawn a vehicle
        vehicle = Vehicle(self.sm, vehicle_type='model3')

        # Spawn a camera sensor and attach it to the vehicle
        # def camera_callback(image):
        #     # Convert image to OpenCV format
        #     frame = Camera.process_data(image)
        #     print(frame.shape)
        #     cv2.imshow("", frame)
        #     cv2.waitKey(1)
        #     # print(f"Camera captured an image at timestamp {image.timestamp}")

        # vehicle.attach_sensor(Camera, callback=camera_callback, transform=carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15)))

        # Spawn a depth camera sensor and attach it to the vehicle
        def depth_camera_callback(image):
            # Convert image to OpenCV format
            frame = DepthCamera.process_data(image)
            print(frame.shape)
            cv2.imshow("", frame)
            cv2.waitKey(1)
            # print(f"Camera captured an image at timestamp {image.timestamp}")

        vehicle.attach_sensor(DepthCamera, callback=depth_camera_callback, transform=carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15)))

        # Spawn a lidar sensor and attach it to the vehicle
        # def lidar_callback(data):
        #     # Convert data to OpenCV format
        #     frame = Lidar.process_data(data)
        #     # print(frame.shape)
        #     # print(f"Lidar captured data at timestamp {data.timestamp}")

        # self.sm.spawn_sensor(Lidar, vehicle=vehicle, callback=lidar_callback, transform=carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15)))

        # Move the spectator to the vehicle
        self.sm.move_spectator(vehicle.get_actor())

        # Run the simulation for a certain duration
        simulation_duration = 60  # seconds
        start_time = time.time()
        while time.time() - start_time < simulation_duration:
            # Apply control to the vehicle (e.g., throttle, steering, etc.)
            vehicle.apply_control(throttle=0.5)
            time.sleep(0.1)
    
    def cleanup(self):
        """
        Cleans up the simulation.
        """
        self.sm.destroy()
