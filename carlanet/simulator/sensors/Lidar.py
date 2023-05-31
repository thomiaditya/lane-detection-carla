import carla
import numpy as np
from .SensorInterface import SensorInterface

class Lidar(SensorInterface):
    """
    Implementation of a Lidar sensor.
    """

    def __init__(self, world: carla.World, blueprint_library: carla.BlueprintLibrary, vehicle: carla.Vehicle):
        """
        Initializes the Lidar sensor.

        Parameters:
            world (carla.World): The CARLA world object.
            blueprint_library (carla.BlueprintLibrary): The blueprint library.
            vehicle (carla.Vehicle): The vehicle to attach the sensor to.
        """
        self.world = world
        self.vehicle = vehicle
        self.blueprint = None
        self.sensor_actor = None

        # Find the blueprint for the lidar sensor.
        self.blueprint = blueprint_library.find('sensor.lidar.ray_cast')

        # Modify the attributes of the blueprint to set the number of points per second, rotation frequency, range, etc.
        self.blueprint.set_attribute('channels', '32')
        self.blueprint.set_attribute('points_per_second', '100000')
        self.blueprint.set_attribute('rotation_frequency', '10')
        self.blueprint.set_attribute('range', '30')

    def setup_sensor(self, transform: carla.Transform):
        """
        Set up the lidar sensor with a specific transform.

        Parameters:
            transform (carla.Transform): The transform to place the sensor.
        """
        self.sensor_actor = self.world.spawn_actor(self.blueprint, transform, attach_to=self.vehicle)

    def process_data(carla_data: carla.LidarMeasurement):
        """
        Process the raw lidar data.

        Parameters:
            raw_data (carla.LidarMeasurement): The raw data from the sensor.

        Returns:
            Processed lidar data.
        """
        # Get the raw data
        print(len(carla_data))

        return 0

    def set_blueprint_attribute(self, attribute: str, value: str):
        """
        Set an attribute of the blueprint.

        Args:
            attribute (str): The name of the attribute.
            value (str): The value to set the attribute to.
        """
        self.blueprint.set_attribute(attribute, value)

    def get_actor(self):
        """
        Get the sensor object.

        Returns:
            The sensor object.
        """
        return self.sensor_actor

    def listen(self, callback: callable):
        """
        Set a callback function to be called every time the sensor receives data.

        Parameters:
            callback (function): The callback function. Should accept one argument - the data from the sensor.
        """
        self.sensor_actor.listen(callback)

    def destroy(self):
        """
        Clean up the sensor, stop it, and detach it from the vehicle.
        """
        self.sensor_actor.stop()
        self.sensor_actor.destroy()
