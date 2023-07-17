import random
import carla
import numpy as np
from ..sensors.SensorInterface import SensorInterface
from ..SimulatorManager import SimulatorManager

class Vehicle:
    """
    A class to manage vehicle creation, sensor attachment, and vehicle management.
    """

    def __init__(self, simulator_manager: SimulatorManager, vehicle_type: str = 'model3', spawn_point: carla.Transform = None):
        """
        Initializes the vehicle.

        Parameters:
            simulator_manager (SimulatorManager): The simulator manager to use.
            vehicle_type (str): The type of vehicle to spawn. Defaults to 'model3'.
            spawn_point (carla.Transform): The spawn point of the vehicle. Defaults to None.
        """
        # Attributes
        self.simulator_manager = simulator_manager
        self.world = simulator_manager.get_world()
        self.blueprint_library = simulator_manager.get_blueprint_library()
        self.vehicle_actor = None
        self.sensors = {}
        self.max_speed = 60.0
        self.follow_vehicle = False

        # Spawn vehicle
        vehicle_bp = self.blueprint_library.filter(vehicle_type)[0]
        if spawn_point is None:
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = spawn_points[0] if spawn_points else carla.Transform()
            # spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()

        self.vehicle_actor = self.world.spawn_actor(vehicle_bp, spawn_point)
        print(f"Vehicle spawned at {spawn_point}.")

        if self.simulator_manager.check_sync_mode():
            self.world.tick()

        # Add the actor to the list of actors in the simulator manager
        self.simulator_manager.add_actor(self.vehicle_actor)

    def get_state(self):
        """
        Returns the state of the vehicle.

        Returns:
            dict: The state of the vehicle.
        """
        state = {}
        state['x'] = self.vehicle_actor.get_transform().location.x
        state['y'] = self.vehicle_actor.get_transform().location.y
        state['psi'] = self.get_yaw()
        state['v'] = self.get_velocity()
        return state
    
    def set_autopilot(self, enabled: bool, traffic_manager_port: int = 8000):
        """
        Sets the autopilot of the vehicle.

        Parameters:
            enabled (bool): Whether to enable the autopilot.
            traffic_manager_port (int): The port of the traffic manager. Defaults to 8000.
        """
        self.vehicle_actor.set_autopilot(enabled, traffic_manager_port)

    def attach_sensor(self, sensor: SensorInterface, transform: carla.Transform = None, callback: callable = None):
        """
        Attaches a sensor to the vehicle.

        Parameters:
            sensor (SensorInterface): The sensor to attach.
            transform (carla.Transform): The transform to attach the sensor to. Defaults to None.
            callback (callable): The callback function to call when the sensor captures data. Defaults to None.
        """
        sensor = sensor(self.world, self.blueprint_library, self.vehicle_actor)

        # Set up sensor
        if transform is None:
            sensor_location = carla.Location(x=1.5, z=2.4)
            sensor_rotation = carla.Rotation(pitch=-15)
            transform = carla.Transform(sensor_location, sensor_rotation)

        sensor.setup_sensor(transform)

        # Add sensor to list of sensors
        self.sensors[str(sensor.get_actor().type_id) + '_' + str(sensor.get_actor().id)] = sensor
        self.simulator_manager.add_actor(sensor.get_actor())

        # Listen to sensor
        if callback is not None:
            sensor.listen(callback)
    
    def get_velocity(self):
        """
        Returns the velocity of the vehicle.

        Returns:
            float: The velocity of the vehicle.
        """
        velocity = self.vehicle_actor.get_velocity()
        return np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    
    def get_yaw(self):
        """
        Returns the yaw angle of the vehicle.

        Returns:
            float: The yaw angle of the vehicle.
        """
        yaw_deg = self.vehicle_actor.get_transform().rotation.yaw

        return np.radians(yaw_deg)
    
    def get_location(self):
        """
        Returns the location of the vehicle.

        Returns:
            carla.Location: The location of the vehicle.
        """
        return self.vehicle_actor.get_transform().location
    
    def set_follow_vehicle(self, follow_vehicle: bool):
        """
        Sets whether the spectator should follow the vehicle.

        Parameters:
            follow_vehicle (bool): Whether to follow the vehicle.
        """
        self.follow_vehicle = follow_vehicle

    def apply_control(self, throttle: float = 0.0, steer: float = 0.0, brake: float = 0.0, reverse: bool = False, hand_brake: bool = False, manual_gear_shift: bool = False, gear: int = 1, vehicle_control: carla.VehicleControl = None):
        """
        Applies control to the vehicle.

        Parameters:
            throttle (float): The throttle value between 0 and 1.
            steer (float): The steer value between -1 and 1.
            brake (float): The brake value between 0 and 1.
            reverse (bool): Whether to reverse the vehicle. Defaults to False.
            hand_brake (bool): Whether to apply the hand brake. Defaults to False.
            manual_gear_shift (bool): Whether to manually shift gears. Defaults to False.
            gear (int): The gear to shift to. Defaults to 1.
            vehicle_control (carla.VehicleControl): The vehicle control command to apply. Defaults to None.
        """
        if vehicle_control is None:
            control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake, reverse=reverse, hand_brake=hand_brake, manual_gear_shift=manual_gear_shift, gear=gear)
        else:
            control = vehicle_control
        self.vehicle_actor.apply_control(control)

        if self.follow_vehicle:
            self.simulator_manager.move_spectator(self.vehicle_actor)
    
    def get_sensors(self):
        """
        Returns the sensors attached to the vehicle.

        Returns:
            dict: The sensors attached to the vehicle.
        """
        return self.sensors
    
    def get_actor(self):
        """
        Returns the vehicle actor.

        Returns:
            carla.Vehicle: The vehicle actor.
        """
        return self.vehicle_actor
