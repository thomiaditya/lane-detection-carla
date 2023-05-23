import carla
import time
import cv2
import numpy as np
from .simulator.SimulatorManager import SimulatorManager
from .simulator.sensors.Camera import Camera
from .simulator.sensors.DepthCamera import DepthCamera
from .simulator.sensors.Lidar import Lidar

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

    def test_run(self):
        """
        Runs the test application flow.
        """
        # Spawn a vehicle
        vehicle = self.sm.spawn_vehicle()

        # Spawn a camera sensor and attach it to the vehicle
        # def camera_callback(image):
        #     # Convert image to OpenCV format
        #     frame = Camera.process_data(image)
        #     print(frame.shape)
        #     cv2.imshow("", frame)
        #     cv2.waitKey(1)
        #     # print(f"Camera captured an image at timestamp {image.timestamp}")

        # self.sm.spawn_sensor(Camera, vehicle=vehicle, callback=camera_callback, transform=carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15)))

        # Spawn a depth camera sensor and attach it to the vehicle
        # def depth_camera_callback(image):
        #     # Convert image to OpenCV format
        #     frame = DepthCamera.process_data(image)
        #     print(frame.shape)
        #     cv2.imshow("", frame)
        #     cv2.waitKey(1)
        #     # print(f"Camera captured an image at timestamp {image.timestamp}")

        # self.sm.spawn_sensor(DepthCamera, vehicle=vehicle, callback=depth_camera_callback, transform=carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15)))

        # Spawn a lidar sensor and attach it to the vehicle
        def lidar_callback(data):
            # Convert data to OpenCV format
            frame = Lidar.process_data(data)
            # print(frame.shape)
            # print(f"Lidar captured data at timestamp {data.timestamp}")

        self.sm.spawn_sensor(Lidar, vehicle=vehicle, callback=lidar_callback, transform=carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15)))

        # Move the spectator to the vehicle
        self.sm.move_spectator(vehicle)

        # Run the simulation for a certain duration
        simulation_duration = 10  # seconds
        start_time = time.time()
        while time.time() - start_time < simulation_duration:
            # Apply control to the vehicle (e.g., throttle, steering, etc.)
            self.sm.apply_control(vehicle, throttle=0.5)
            time.sleep(0.1)

        # Cleanup
        self.handle_cleanup()
    
    def handle_cleanup(self):
        """
        Cleans up the simulation.
        """
        # If user presses Ctrl+C, the simulation will be cleaned up
        if KeyboardInterrupt:
            self.sm.destroy()
