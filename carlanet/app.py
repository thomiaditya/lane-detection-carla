import random
import carla
import time
import cv2
import numpy as np
import functools
import os
import os.path as osp
import wandb
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
from .simulator.CLRNet import CLRNet
from .simulator.vehicle.MPC import ModelPredictiveController

from queue import Queue

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
        self.world = self.sm.get_world()
        self.wandb_instance = None

        self.frame_times = []
        self.vehicle_list = []

        self.sm.set_sync_mode(True)

    def set_simulator_weather(self, weather_carla):
        """
        Sets the weather of the CARLA simulator.

        Args:
            weather (carla.WeatherParameters): The weather to set.
        """
        weather = self.world.get_weather()
        weather.sun_altitude_angle = -30
        weather.fog_density = 65
        weather.fog_distance = 10
        self.world.set_weather(weather_carla)
    
    def setup_wandb(self, experiment_name, tags=[]):
        """
        Set up Weights and Biases.

        Args:
            project_name (str): The name of the project.
            experiment_name (str): The name of the experiment.
        """
        return wandb.init(project="lane-detection-carla", name=experiment_name, tags=tags, reinit=True)
    
    @handle_exception
    def run_yolo(self):
        """
        Runs the main application flow.
        """
        print("Preparing the YOLOPv2 model...")
        self.detector = YOLOPv2(device='cuda')
        print("YOLOPv2 model ready")

        # List the weathers
        weathers = [
            # Clear
            [carla.WeatherParameters.ClearSunset, "clear_sunset"],
            [carla.WeatherParameters.ClearNight, "clear_night"],
            # Rain
            [carla.WeatherParameters.HardRainSunset, "hard_rain_sunset"],
            [carla.WeatherParameters.HardRainNight, "hard_rain_night"],
            # Cloudy
            [carla.WeatherParameters.CloudySunset, "cloudy_sunset"],
            [carla.WeatherParameters.CloudyNight, "cloudy_night"],
            # Wet
            [carla.WeatherParameters.WetSunset, "wet_sunset"],
            [carla.WeatherParameters.WetNight, "wet_night"],
        ]

        # Setup wandb
        # FIXME: Change the experiment name

        # Loop through the weathers
        for weather in weathers:
            wandb = self.setup_wandb(f"yolopv2-mpc-{weather[1]}", tags=[weather[1]])
            # Set the weather
            self.set_simulator_weather(weather[0])
            print(f"Running the simulation with weather {weather[1]}...")

            self.vehicle = Vehicle(self.sm, vehicle_type='model3')

            sensor_queue = Queue()
            
            self.vehicle.attach_sensor(Camera, transform=carla.Transform(carla.Location(x=1.5, z=2.4)), callback=self.camera_callback(sensor_queue))
            print("Camera attached")

            # Move the spectator to the vehicle
            self.sm.move_spectator(self.vehicle.get_actor())
            self.vehicle.set_follow_vehicle(True)

            # =========================================================================================
            # Previous variables
            # =========================================================================================
            t_prev = 0
            v_prev = 0
            a = 0
            delta = 0

            # =========================================================================================
            # Implementing MPC
            # =========================================================================================
            
            dt = 0.05
            N = 20
            L = self.vehicle.get_actor().bounding_box.extent.x * 2

            # Test the MPC
            mpc = ModelPredictiveController(dt=dt, N=N, L=L)
            print(mpc.mpc)

            # Generate waypoints
            waypoints = self.sm.generate_example_waypoints(self.vehicle.get_location(), 1, 500)

            # Start at the first waypoint
            current_waypoint = waypoints[:N]

            # Run loop only 200 frames
            frames = 500
            for i in range(frames):
                # time.sleep(0.1)  # Sleep for 0.05 seconds (i.e., 20Hz update rate)
                # Tick the CARLA simulator
                if self.sm.check_sync_mode(): self.sm.tick()

                t = time.time()
                start_time = t
                v = self.vehicle.get_velocity()
                x = self.vehicle.get_location().x
                y = self.vehicle.get_location().y
                yaw = self.vehicle.get_yaw()
                STEP_TIME = t - t_prev

                # Update the current waypoint
                closest_waypoint_index = self.find_closest_waypoint_index(waypoints)
                current_waypoint = waypoints[closest_waypoint_index:closest_waypoint_index + N]

                # self.debug_waypoints(current_waypoint) # Debug the waypoints

                converted_waypoints = self.map_to_local(x, y, yaw, current_waypoint) # Convert the waypoints to the vehicle reference frame

                coeffs = mpc.get_coeffs(converted_waypoints) # Get the coefficients of the polynomial
                mpc.update_coeff(coeffs) # Update the coefficients of the polynomial

                # Get desired speed using curvature formula
                k = self.curvature(coeffs, 20) # Get the curvature at x
                v_des = self.calculate_speed(k, 30, 10) # Calculate the desired speed
                mpc.set_desired_speed(v_des) # Set the desired speed

                # Debug the desired speed using string
                self.sm.get_world().debug.draw_string(self.vehicle.get_location() + carla.Location(x=0, y=0, z=1), 
                                                    f'Speed: {v:.2f}/{v_des:.2f}',
                                    draw_shadow=False, color=carla.Color(0, 255, 255), life_time=dt, persistent_lines=False)

                # Set initial states
                x0 = np.array([0, 0, 0, v, 0, 0])
                x0 = mpc.get_initial_state(x0, a, delta, STEP_TIME, coeffs)

                mpc.set_init_guess(x0, u0 = np.array([a, delta]))

                # Get the control commands
                u0 = mpc.step(x0)[:, 0]

                # Apply control to the vehicle
                a = u0[0]
                delta = u0[1]

                control = self.set_control(a, delta)
                self.vehicle.apply_control(vehicle_control=control)


                # =========================================================================================
                # Lane detector
                # =========================================================================================
                # Wait for the camera to process an image
                image = sensor_queue.get()

                preds = self.detector.predict(image)
                detected_lane = self.detector.show_detection(preds)

                end_time = time.time()

                self.frame_times.append(end_time - start_time)

                if len(self.frame_times) > 100:
                    self.frame_times.pop(0)

                fps = 0
                if len(self.frame_times) > 0:
                    avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                    fps = 1.0 / avg_frame_time
                    # Delete line
                    print('FPS: ', fps, end='\r')
                
                # Log the metrics to wandb
                wandb.log({
                    f"fps": fps,
                    f"speed": v,
                    f"desired_speed": v_des,
                }, step=i)

                # =========================================================================================
                # End of lane detector
                # =========================================================================================
                
                t_prev = t
                v_prev = v

            # =========================================================================================
            # Calculate the FPS
            # =========================================================================================
            fps_file = f"fps-yolo.txt"
            if len(self.frame_times) > 0:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                fps = 1.0 / avg_frame_time
                # Write the FPS to a file for every weather
                with open(fps_file, 'a') as f:
                    f.write(f"{weather[1]}: {fps:.4f}\n")
                
                print(f"Average FPS: {fps:.2f}. Already written to {fps_file}.")

                # Reset the frame times
                self.frame_times = []

            self.cleanup()
    
    @handle_exception
    def run_clr(self):
        """
        Runs the main application flow.
        """
        current_file_path = os.path.dirname(os.path.realpath(__file__))

        # Get project root path from current file path. Keep going up one directory until we find the "setup.py" file.
        while not os.path.exists(os.path.join(current_file_path, "setup.py")):
            current_file_path = os.path.dirname(current_file_path)

        print("Preparing the CLRNet model...")
        self.detector = CLRNet(config_path=osp.join(current_file_path, "model", "clr_resnet34_culane.py"), weight_path=osp.join(current_file_path, "model", "culane_r34.pth"))
        print("CLRNet model ready")

        # List the weathers
        weathers = [
            # Clear
            [carla.WeatherParameters.ClearSunset, "ClearSunset"],
            [carla.WeatherParameters.ClearNight, "ClearNight"],
            # Rain
            [carla.WeatherParameters.HardRainSunset, "HardRainSunset"],
            [carla.WeatherParameters.HardRainNight, "HardRainNight"],
            # Cloudy
            [carla.WeatherParameters.CloudySunset, "CloudySunset"],
            [carla.WeatherParameters.CloudyNight, "CloudyNight"],
            # Wet
            [carla.WeatherParameters.WetSunset, "WetSunset"],
            [carla.WeatherParameters.WetNight, "WetNight"],
        ]

        # Loop through the weathers
        for weather in weathers:
            # Set the weather
            self.set_simulator_weather(weather[0])
            print(f"Running the simulation with weather {weather[1]}...")
            self.vehicle = Vehicle(self.sm, vehicle_type='model3')

            sensor_queue = Queue()
            
            self.vehicle.attach_sensor(Camera, transform=carla.Transform(carla.Location(x=1.5, z=2.4)), callback=self.camera_callback(sensor_queue))
            print("Camera attached")

            # Move the spectator to the vehicle
            self.sm.move_spectator(self.vehicle.get_actor())
            self.vehicle.set_follow_vehicle(True)

            # =========================================================================================
            # Previous variables
            # =========================================================================================
            t_prev = 0
            v_prev = 0
            a = 0
            delta = 0

            # =========================================================================================
            # Implementing MPC
            # =========================================================================================
            
            dt = 0.05
            N = 20
            L = self.vehicle.get_actor().bounding_box.extent.x * 2

            # Test the MPC
            mpc = ModelPredictiveController(dt=dt, N=N, L=L)
            print(mpc.mpc)

            # Generate waypoints
            waypoints = self.sm.generate_example_waypoints(self.vehicle.get_location(), 1, 500)

            # Start at the first waypoint
            current_waypoint = waypoints[:N]

            # # Live plot
            # plt.ion()
            # fig, axs = plt.subplots()
            # x = np.linspace(0, 20)
            # y = np.linspace(-15, 15)
            # wp, = axs.plot(x, y, 'b-')
            # wp_ref, = axs.plot(x, y, 'r-')
            
            # Run loop only 200 frames
            frames = 500
            for i in range(frames):
                # time.sleep(0.1)  # Sleep for 0.05 seconds (i.e., 20Hz update rate)
                # Tick the CARLA simulator
                if self.sm.check_sync_mode(): self.sm.tick()

                t = time.time()
                start_time = t
                v = self.vehicle.get_velocity()
                x = self.vehicle.get_location().x
                y = self.vehicle.get_location().y
                yaw = self.vehicle.get_yaw()
                STEP_TIME = t - t_prev

                # Update the current waypoint
                closest_waypoint_index = self.find_closest_waypoint_index(waypoints)
                current_waypoint = waypoints[closest_waypoint_index:closest_waypoint_index + N]

                # self.debug_waypoints(current_waypoint) # Debug the waypoints

                converted_waypoints = self.map_to_local(x, y, yaw, current_waypoint) # Convert the waypoints to the vehicle reference frame

                coeffs = mpc.get_coeffs(converted_waypoints) # Get the coefficients of the polynomial
                mpc.update_coeff(coeffs) # Update the coefficients of the polynomial

                # Get desired speed using curvature formula
                k = self.curvature(coeffs, 20) # Get the curvature at x
                v_des = self.calculate_speed(k, 30, 10) # Calculate the desired speed
                mpc.set_desired_speed(v_des) # Set the desired speed

                # Debug the desired speed using string
                self.sm.get_world().debug.draw_string(self.vehicle.get_location() + carla.Location(x=0, y=0, z=1), 
                                                    f'Speed: {v:.2f}/{v_des:.2f}',
                                    draw_shadow=False, color=carla.Color(0, 255, 255), life_time=dt, persistent_lines=False)

                # Set initial states
                x0 = np.array([0, 0, 0, v, 0, 0])
                x0 = mpc.get_initial_state(x0, a, delta, STEP_TIME, coeffs)

                mpc.set_init_guess(x0, u0 = np.array([a, delta]))

                # Get the control commands
                u0 = mpc.step(x0)[:, 0]

                # Apply control to the vehicle
                a = u0[0]
                delta = u0[1]

                control = self.set_control(a, delta)
                self.vehicle.apply_control(vehicle_control=control)

                # print(control) # Print the control commands

                # # Plot the waypoints with x coordinates as the x axis and y coordinates as the y axis
                # wp.set_xdata(converted_waypoints[0])
                # wp.set_ydata(converted_waypoints[1])

                # wp_ref.set_xdata(converted_waypoints[0])
                # wp_ref.set_ydata(np.polyval(coeffs, converted_waypoints[0]))

                # # axs.relim()
                # axs.autoscale_view(scalex=True, scaley=False)

                # plt.draw()
                # plt.pause(0.001)

                # =========================================================================================
                # Lane detector
                # =========================================================================================
                # Wait for the camera to process an image
                image = sensor_queue.get()

                preds = self.detector.run(image)

                end_time = time.time()

                self.frame_times.append(end_time - start_time)

                if len(self.frame_times) > 100:
                    self.frame_times.pop(0)

                if len(self.frame_times) > 0:
                    avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                    fps = 1.0 / avg_frame_time
                    # Delete line
                    print('FPS: ', fps, end='\r')

                # =========================================================================================
                # End of lane detector
                # =========================================================================================
                
                t_prev = t
                v_prev = v

            # =========================================================================================
            # Calculate the FPS
            # =========================================================================================
            fps_file = f"{current_file_path}/fps-clrnet.txt"
            if len(self.frame_times) > 0:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                fps = 1.0 / avg_frame_time
                # Write the FPS to a file for every weather
                with open(fps_file, 'a') as f:
                    f.write(f"{weather[1]}: {fps:.4f}\n")
                
                print(f"Average FPS: {fps:.2f}. Already written to {fps_file}.")

            self.cleanup()

    @handle_exception
    def run_traffic_yolo(self):
        print("Preparing the YOLOPv2 model...")
        self.detector = YOLOPv2(device='cuda')
        print("YOLOPv2 model ready")

        # Run the simulation for a while to let the vehicles drive around.
        # List the weathers
        weathers = [
            # Clear
            [carla.WeatherParameters.ClearSunset, "ClearSunset"],
            [carla.WeatherParameters.ClearNight, "ClearNight"],
            # Rain
            [carla.WeatherParameters.HardRainSunset, "HardRainSunset"],
            [carla.WeatherParameters.HardRainNight, "HardRainNight"],
            # Cloudy
            [carla.WeatherParameters.CloudySunset, "CloudySunset"],
            [carla.WeatherParameters.CloudyNight, "CloudyNight"],
            # Wet
            [carla.WeatherParameters.WetSunset, "WetSunset"],
            [carla.WeatherParameters.WetNight, "WetNight"],
        ]

        # Loop through the weathers
        for weather in weathers:
            # Set the weather
            self.set_simulator_weather(weather[0])
            print(f"Running the simulation with weather {weather[1]}...")

            traffic_manager = self.sm.traffic_manager

            tm_port = traffic_manager.get_port()

            # Get the blueprint for a random vehicle.
            blueprints = self.world.get_blueprint_library().filter('vehicle.*')
            blueprints = [x for x in blueprints if not x.id.endswith('bicycle')]

            # Spawn main vehicle
            self.vehicle = Vehicle(self.sm, vehicle_type='model3')

            sensor_queue = Queue()
            
            self.vehicle.attach_sensor(Camera, transform=carla.Transform(carla.Location(x=1.5, z=2.4)), callback=self.camera_callback(sensor_queue))
            print("Camera attached")

            # Move the spectator to the vehicle
            self.sm.move_spectator(self.vehicle.get_actor())
            self.vehicle.set_follow_vehicle(True)
            self.vehicle.set_autopilot(True, tm_port)

            # Create a list to hold all our vehicles (so we can clean up properly later on).
            spawn_points = self.world.get_map().get_spawn_points()
            for _ in range(50):
                blueprint = random.choice(blueprints)
                # Spawn a vehicle.
                for _ in range(10):  # Try up to 10 times.
                    try:
                        spawn_point = random.choice(spawn_points)
                        vehicle = self.world.spawn_actor(blueprint, spawn_point)
                        self.vehicle_list.append(vehicle)
                        # Set the vehicle to autopilot using Traffic Manager.
                        vehicle.set_autopilot(True, tm_port)
                        self.world.tick()  # Manually advance time in synchronous mode.
                        break  # Success, so break out of the loop.
                    except RuntimeError:
                        continue  # Try again with a different spawn point.

            print('Spawned %d vehicles.' % len(self.vehicle_list))

            frames = 500
            for _ in range(frames):
                self.sm.tick()
                t = time.time()
                start_time = t

                self.sm.move_spectator(self.vehicle.get_actor())

                # =========================================================================================
                # Lane detector
                # =========================================================================================
                # Wait for the camera to process an image
                image = sensor_queue.get()

                preds = self.detector.predict(image)
                detected_lane = self.detector.show_detection(preds)

                end_time = time.time()

                self.frame_times.append(end_time - start_time)

                if len(self.frame_times) > 100:
                    self.frame_times.pop(0)

                if len(self.frame_times) > 0:
                    avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                    fps = 1.0 / avg_frame_time
                    # Delete line
                    print('FPS: ', fps, end='\r')

            # =========================================================================================
            # Calculate the FPS
            # =========================================================================================
            fps_file = f"fps-yolo-traffic.txt"
            if len(self.frame_times) > 0:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                fps = 1.0 / avg_frame_time
                # Write the FPS to a file for every weather
                with open(fps_file, 'a') as f:
                    f.write(f"{weather[1]}: {fps:.4f}\n")
                
                print(f"Average FPS: {fps:.2f}. Already written to {fps_file}.")

                # Reset the frame times
                self.frame_times = []

            # Clean up all vehicles.
            self.cleanup()

    @handle_exception
    def run_traffic_clrnet(self):
        current_file_path = os.path.dirname(os.path.realpath(__file__))

        # Get project root path from current file path. Keep going up one directory until we find the "setup.py" file.
        while not os.path.exists(os.path.join(current_file_path, "setup.py")):
            current_file_path = os.path.dirname(current_file_path)

        print("Preparing the CLRNet model...")
        self.detector = CLRNet(config_path=osp.join(current_file_path, "model", "clr_resnet34_culane.py"), weight_path=osp.join(current_file_path, "model", "culane_r34.pth"))
        print("CLRNet model ready")

        # Run the simulation for a while to let the vehicles drive around.
        # List the weathers
        weathers = [
            # Clear
            [carla.WeatherParameters.ClearSunset, "ClearSunset"],
            [carla.WeatherParameters.ClearNight, "ClearNight"],
            # Rain
            [carla.WeatherParameters.HardRainSunset, "HardRainSunset"],
            [carla.WeatherParameters.HardRainNight, "HardRainNight"],
            # Cloudy
            [carla.WeatherParameters.CloudySunset, "CloudySunset"],
            [carla.WeatherParameters.CloudyNight, "CloudyNight"],
            # Wet
            [carla.WeatherParameters.WetSunset, "WetSunset"],
            [carla.WeatherParameters.WetNight, "WetNight"],
        ]

        # Loop through the weathers
        for weather in weathers:
            # Set the weather
            self.set_simulator_weather(weather[0])
            print(f"Running the simulation with weather {weather[1]}...")

            traffic_manager = self.sm.traffic_manager

            tm_port = traffic_manager.get_port()

            # Get the blueprint for a random vehicle.
            blueprints = self.world.get_blueprint_library().filter('vehicle.*')
            blueprints = [x for x in blueprints if not x.id.endswith('bicycle')]

            # Spawn main vehicle
            self.vehicle = Vehicle(self.sm, vehicle_type='model3')

            sensor_queue = Queue()
            
            self.vehicle.attach_sensor(Camera, transform=carla.Transform(carla.Location(x=1.5, z=2.4)), callback=self.camera_callback(sensor_queue))
            print("Camera attached")

            # Move the spectator to the vehicle
            self.sm.move_spectator(self.vehicle.get_actor())
            self.vehicle.set_follow_vehicle(True)
            self.vehicle.set_autopilot(True, tm_port)

            # Create a list to hold all our vehicles (so we can clean up properly later on).
            spawn_points = self.world.get_map().get_spawn_points()
            for _ in range(50):
                blueprint = random.choice(blueprints)
                # Spawn a vehicle.
                for _ in range(10):  # Try up to 10 times.
                    try:
                        spawn_point = random.choice(spawn_points)
                        vehicle = self.world.spawn_actor(blueprint, spawn_point)
                        self.vehicle_list.append(vehicle)
                        # Set the vehicle to autopilot using Traffic Manager.
                        vehicle.set_autopilot(True, tm_port)
                        self.world.tick()  # Manually advance time in synchronous mode.
                        break  # Success, so break out of the loop.
                    except RuntimeError:
                        continue  # Try again with a different spawn point.

            print('Spawned %d vehicles.' % len(self.vehicle_list))

            frames = 500
            for i in range(frames):
                self.sm.tick()
                t = time.time()
                start_time = t

                self.sm.move_spectator(self.vehicle.get_actor())

                # =========================================================================================
                # Lane detector
                # =========================================================================================
                # Wait for the camera to process an image
                image = sensor_queue.get()

                preds = self.detector.run(image)

                end_time = time.time()

                self.frame_times.append(end_time - start_time)

                if len(self.frame_times) > 100:
                    self.frame_times.pop(0)

                if len(self.frame_times) > 0:
                    avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                    fps = 1.0 / avg_frame_time
                    # Delete line
                    print(f'{i} FPS: ', fps, end='\r')

            # =========================================================================================
            # Calculate the FPS
            # =========================================================================================
            fps_file = f"fps-clrnet-traffic.txt"
            if len(self.frame_times) > 0:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                fps = 1.0 / avg_frame_time
                # Write the FPS to a file for every weather
                with open(fps_file, 'a') as f:
                    f.write(f"{weather[1]}: {fps:.4f}\n")
                
                print(f"Average FPS: {fps:.2f}. Already written to {fps_file}.")

                # Reset the frame times
                self.frame_times = []

            # Clean up all vehicles.
            self.cleanup()

    def calculate_speed(self, curvature, max_speed, beta):
        """
        Calculate the desired speed based on curvature.
        
        :param curvature: The curvature at a specific point along the path.
        :param max_speed: The maximum speed of the vehicle.
        :param beta: A tunable parameter that determines how much the speed decreases with increasing curvature.
        
        :return: The desired speed at the given curvature.
        """
        speed = max_speed / (1 + beta * abs(curvature))
        return speed

    def curvature(self, poly_coeffs, x):
        """
        Compute the curvature of a polynomial at a specific x.
        
        :param poly_coeffs: The coefficients of the polynomial, in decreasing order.
        :param x: The x-value at which to compute the curvature.
        
        :return: The curvature at x.
        """
        # Calculate the first derivative
        first_derivative_coeffs = np.polyder(poly_coeffs)
        first_derivative = np.polyval(first_derivative_coeffs, x)

        # Calculate the second derivative
        second_derivative_coeffs = np.polyder(first_derivative_coeffs)
        second_derivative = np.polyval(second_derivative_coeffs, x)

        # Calculate the curvature
        curvature = second_derivative / (1 + first_derivative**2)**1.5

        return curvature

    def set_control(self, a, delta):
        """
        Set the control of the vehicle.

        Args:
            a (float): Throttle.
            delta (float): Steering angle.
        """
        control = carla.VehicleControl()

        threshold = 0
        control.throttle = a if a > 0 else 0
        control.brake = a if a < 0 else 0
        control.steer = -delta

        return control

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
        return closest_waypoint_index + 2

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
        debug = self.world.debug

        for i in range(len(waypoints) - 1):
            # Draw a line between each pair of waypoints
            debug.draw_line(waypoints[i].transform.location, waypoints[i + 1].transform.location, 
                            thickness=0.2, color=carla.Color(255, 0, 0), persistent_lines=False, life_time=0.1)

    def camera_callback(self, sensor_queue=None):
        # Create function to handle the camera data
        # Create window to display the camera image
        # cv2.namedWindow("Camera", cv2.WINDOW_AUTOSIZE)
        def callback(image):
            # Change image into a numpy array
            image = Camera.process_data(image)

            if sensor_queue is not None:
                sensor_queue.put(image)

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

        if self.vehicle_list is not None:
            print("Cleaning up vehicles...")
            # Destroy all the vehicles
            for vehicle in self.vehicle_list:
                try:
                    vehicle.destroy()
                except:
                    pass
            print("Vehicles cleaned up")
            
        # Destroy the vehicle
        try:
            self.sm.destroy()
        except:
            pass

        # Clear all debug lines
        self.sm.get_world().debug.draw_line(carla.Location(), carla.Location(), thickness=0.0, color=carla.Color(0, 0, 0), life_time=0.0, persistent_lines=False)
        plt.ioff()

        # Destroy the window
        # cv2.destroyAllWindows()