import carla
import time
import cv2
import numpy as np
import functools
from scipy.spatial import distance
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from .simulator.SimulatorManager import SimulatorManager
from .simulator.vehicle.Vehicle import Vehicle
from .simulator.sensors.Camera import Camera
from .simulator.sensors.DepthCamera import DepthCamera
from .simulator.sensors.Lidar import Lidar
from .simulator.vehicle.VehicleController import VehicleController
from .simulator.YOLOPv2 import YOLOPv2
from .simulator.vehicle.MPC import ModelPredictiveController

def handle_exception(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            func(self, *args, **kwargs)
            self.cleanup()
        except:
            # Print exception with traceback and red text
            import traceback
            print("\033[91m")
            traceback.print_exc()
            print("\033[0m")

            print("Cleaning up...")
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
        self.detector = None
        self.vehicle = None
        
        self.sm.set_sync_mode(False)
    
    @handle_exception
    def run_mpc(self):
        """
        Runs the main application flow.
        """
        self.vehicle = Vehicle(self.sm, vehicle_type='model3')

        # print("Preparing the YOLOPv2 model...")
        # self.detector = YOLOPv2(device='cuda')
        # print("YOLOPv2 model ready")
        
        # vehicle.attach_sensor(Camera, transform=carla.Transform(carla.Location(x=1.5, z=2.4)), callback=self.camera_callback())
        # print("Camera attached")

        # Move the spectator to the vehicle
        self.sm.move_spectator(self.vehicle.get_actor())
        self.vehicle.set_follow_vehicle(True)

        # =========================================================================================
        # Previous variables
        # =========================================================================================
        t_prev = 0
        v_prev = 0
        throttle = 0
        steer = 0

        # =========================================================================================
        # Implementing MPC
        # =========================================================================================
        
        dt = 0.05
        N = 15
        L = self.vehicle.get_actor().bounding_box.extent.x

        # Test the MPC
        mpc = ModelPredictiveController(dt=dt, N=N, L=L)
        print(mpc.mpc)

        # Generate waypoints
        waypoints = self.sm.generate_example_waypoints(self.vehicle.get_location(), 1, 400)

        # Start at the first waypoint
        current_waypoint = waypoints[:N]

        while True:
            # time.sleep(0.1)  # Sleep for 0.05 seconds (i.e., 20Hz update rate)
            t = time.time()
            v = self.vehicle.get_velocity()
            STEP_TIME = t - t_prev

            # Update the current waypoint
            closest_waypoint_index = self.find_closest_waypoint_index(waypoints)
            current_waypoint = waypoints[closest_waypoint_index:closest_waypoint_index + N]

            # if self.vehicle.get_location().distance(current_waypoint[0].transform.location) > 5.0:
            #     break

            self.debug_waypoints(current_waypoint)

            converted_waypoints = self.map_to_local(self.vehicle.get_location().x, self.vehicle.get_location().y, self.vehicle.get_yaw(), current_waypoint)

            coeffs = mpc.get_coeffs(converted_waypoints)
            mpc.update_coeff(coeffs)

            # Set initial states
            states = self.vehicle.get_state()
            x0 = np.array([0, 0, states['psi'], states['v'], 0, 0])
            x0 = mpc.get_initial_state(x0, throttle, steer, STEP_TIME)

            mpc.set_init_guess(x0)

            # Get the control commands
            u0 = mpc.step(x0)[:, 0]

            # Apply control to the vehicle
            throttle = u0[0]
            steer = -np.clip(u0[1], -1, 1)

            self.vehicle.apply_control(throttle, steer)

            print(f"Throttle: {throttle}, Steer: {steer}")

            # Tick the CARLA simulator
            if self.sm.check_sync_mode(): self.sm.tick()
            
            t_prev = t
            v_prev = v
        
        self.cleanup()
        self.run_mpc()

    def find_closest_waypoint_index(self, waypoints):
        """
        Updates the current waypoints.

        Args:
            waypoints (list): A list of waypoints.
        """
        # Find the closest waypoint to the vehicle with minimum distance 3.0
        closest_waypoint = min(waypoints, key=lambda wp: self.vehicle.get_location().distance(wp.transform.location))

        # Find the index of the closest waypoint
        closest_waypoint_index = waypoints.index(closest_waypoint)

        # Return the index of the closest waypoint
        return closest_waypoint_index + 1

    def map_to_local(self, x, y, yaw, waypoints):  
        # Extract the x and y coordinates of the waypoints
        wps_x = np.array([wp.transform.location.x for wp in waypoints])
        wps_y = np.array([wp.transform.location.y for wp in waypoints])
        # print(yaw)
        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)
                
        wp_vehRef_x = cos_yaw * (wps_x - x) - sin_yaw * (wps_y - y)
        wp_vehRef_y = sin_yaw * (wps_x - x) + cos_yaw * (wps_y - y)

        wp_ = np.array([wp_vehRef_x, wp_vehRef_y])

        return wp_

    def debug_waypoints(self, waypoints):
        """
        Debugs the waypoints by drawing lines between each pair of waypoints and drawing a string with the waypoint number at each waypoint location.

        Args:
            waypoints (list): A list of waypoints.
        """
        debug = self.sm.get_world().debug

        for i in range(len(waypoints) - 1):
            # Draw a line between each pair of waypoints
            debug.draw_line(waypoints[i].transform.location, waypoints[i + 1].transform.location, 
                            thickness=0.2, color=carla.Color(255, 0, 0), persistent_lines=False, life_time=0.5)

    def camera_callback(self):
        # Create function to handle the camera data
        def callback(image):
            # Create window to display the camera image
            cv2.namedWindow("Camera", cv2.WINDOW_AUTOSIZE)

            # Change image into a numpy array
            image = Camera.process_data(image)

            # Lane detector
            preds = self.detector.predict(image)
            detected_lane = self.detector.show_detection(preds)
            # Display the detected lane combined with the region of interest
            cv2.imshow("Camera", detected_lane)
            cv2.waitKey(1)

        return callback

    @handle_exception
    def test_run(self):
        """
        Runs the test application flow.
        """
        # Spawn a vehicle
        self.vehicle = Vehicle(self.sm, vehicle_type='model3')

        # Attach a camera sensor to the vehicle
        def callback(image):
            # Change image into a numpy array
            image = Camera.process_data(image)

            # Display the image
            cv2.imshow("Camera", image)
            cv2.waitKey(1)

        self.vehicle.attach_sensor(Camera, transform=carla.Transform(carla.Location(x=1.5, z=2.4)), callback=callback)

        # Create vehicle controller with your chosen parameters
        self.vehicle_controller = VehicleController(Kp_longitudinal=1, Ki_longitudinal=0, Kd_longitudinal=0, Kp_lateral=1e-10)

        # Generate waypoints
        self.waypoints = self.sm.generate_example_waypoints(self.vehicle.get_actor().get_transform().location, 1, 200)

        # Start at the first waypoint
        next_waypoint = self.waypoints[0]

        # Move the spectator to the vehicle
        self.sm.move_spectator(self.vehicle.get_actor())

        # Run the simulation for a certain duration
        while True:
            time.sleep(0.05)  # Sleep for 0.05 seconds (i.e., 20Hz update rate)

            debug = self.sm.get_world().debug

            for i in range(len(self.waypoints) - 1):
                # Draw a line between each pair of waypoints
                debug.draw_line(self.waypoints[i].transform.location, self.waypoints[i + 1].transform.location, 
                                thickness=0.1, color=carla.Color(255, 0, 0), life_time=0.1, persistent_lines=False)

                # Draw a string with the waypoint number at each waypoint location
                debug.draw_string(self.waypoints[i].transform.location, f'WP {i}', 
                                draw_shadow=False, color=carla.Color(0, 0, 255), life_time=0.1, persistent_lines=False) 

            # Apply control to the vehicle (e.g., throttle, steering, etc.)
            current_transform = self.vehicle.get_actor().get_transform()
            current_speed = self.vehicle.get_velocity()
            current_yaw = current_transform.rotation.yaw
            current_position = current_transform.location

            desired_speed = 10.0 # m/s

            # Debug the desired speed using arrows with the desired speed as the length
            debug.draw_arrow(current_transform.location, current_transform.location + carla.Vector3D(desired_speed * np.cos(np.deg2rad(current_yaw)), desired_speed * np.sin(np.deg2rad(current_yaw)), 0), 
                            thickness=0.1, arrow_size=0.1, color=carla.Color(0, 255, 0), life_time=0.1, persistent_lines=False)

            # Get the control commands
            vehicle_control = self.vehicle_controller.control(
                desired_speed,
                current_speed,
                current_position,
                current_yaw,
                next_waypoint,
                dt=0.05  # Assuming a control rate of 20Hz (i.e., a control command every 0.05 seconds)
            )

            # Apply control to the vehicle
            self.vehicle.apply_control(vehicle_control=vehicle_control)

            # Move on to the next waypoint if we're close enough to the current one
            if self.vehicle.get_actor().get_transform().location.distance(next_waypoint.transform.location) < 1.0:
                if len(self.waypoints) > 0:
                    next_waypoint = self.waypoints.pop(0)
            
            print(vehicle_control)

            self.sm.move_spectator(self.vehicle.get_actor())
            # self.sm.visualize_control(vehicle_control)
            
            # Tick the CARLA simulator
            self.sm.get_world().tick()
    
    def cleanup(self):
        """
        Cleans up the simulation.
        """
        self.sm.destroy()

        # Clear all debug lines
        self.sm.get_world().debug.draw_line(carla.Location(), carla.Location(), thickness=0.0, color=carla.Color(0, 0, 0), life_time=0.0, persistent_lines=False)

        # Destroy the window
        # cv2.destroyAllWindows()