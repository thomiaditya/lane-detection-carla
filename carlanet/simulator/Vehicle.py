import random
import carla
from .SensorInterface import SensorInterface
from .SimulatorManager import SimulatorManager

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

        # Spawn vehicle
        vehicle_bp = self.blueprint_library.filter(vehicle_type)[0]
        if spawn_point is None:
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()

        self.vehicle_actor = self.world.spawn_actor(vehicle_bp, spawn_point)

        # Add the actor to the list of actors in the simulator manager
        self.simulator_manager.add_actor(self.vehicle_actor)

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

    def apply_control(self, throttle: float = 0.0, steer: float = 0.0, brake: float = 0.0, reverse: bool = False, hand_brake: bool = False, manual_gear_shift: bool = False, gear: int = 1):
        """
        Applies control to the vehicle.

        Parameters:
            throttle (float): The throttle value between 0 and 1.
            steer (float): The steer value between -1 and 1.
            brake (float): The brake value between 0 and 1.
        """
        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake, reverse=reverse, hand_brake=hand_brake, manual_gear_shift=manual_gear_shift, gear=gear)
        self.vehicle_actor.apply_control(control)

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
