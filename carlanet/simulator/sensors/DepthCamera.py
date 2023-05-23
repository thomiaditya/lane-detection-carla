import carla
import numpy as np
from ..SensorInterface import SensorInterface

class DepthCamera(SensorInterface):
    """
    Implementation of a depth camera sensor.
    """

    def __init__(self, world: carla.World, blueprint_library: carla.BlueprintLibrary, vehicle: carla.Vehicle):
        """
        Initializes the depth camera.

        Parameters:
            world (carla.World): The CARLA world object.
            blueprint_library (carla.BlueprintLibrary): The blueprint library.
            vehicle (carla.Vehicle): The vehicle to attach the sensor to.
        """
        self.world = world
        self.vehicle = vehicle
        self.blueprint = None
        self.sensor_actor = None

        # Find the blueprint for the depth camera sensor.
        self.blueprint = blueprint_library.find('sensor.camera.depth')

        # Modify the attributes of the blueprint to set image resolution and field of view.
        self.blueprint.set_attribute('image_size_x', '800')
        self.blueprint.set_attribute('image_size_y', '600')
        self.blueprint.set_attribute('fov', '110')

        # Set the time in seconds between sensor captures
        self.blueprint.set_attribute('sensor_tick', '1.0')

    def setup_sensor(self, transform: carla.Transform):
        """
        Set up the depth camera sensor with a location and rotation.

        Parameters:
            transform (carla.Transform): The transform to place the sensor.
        """
        self.sensor_actor = self.world.spawn_actor(self.blueprint, transform, attach_to=self.vehicle)
    
    def set_blueprint_attribute(self, attribute: str, value: str):
        """
        Set an attribute of the blueprint.

        Args:
            attribute (str): The name of the attribute.
            value (str): The value to set the attribute to.
        """
        self.blueprint.set_attribute(attribute, value)

    def process_data(carla_data: carla.Image):
        """
        Process the raw depth data.

        Parameters:
            raw_data (carla.Image): The raw data from the sensor.

        Returns:
            Processed depth data.
        """
        image = carla_data
        image.convert(carla.ColorConverter.Depth)
        image_data = image.raw_data
        image_size = (image.height, image.width, 4)
        depth_frame = np.frombuffer(image_data, dtype=np.uint8).reshape(image_size)
        depth_frame = depth_frame[:, :, :3]

        return depth_frame
    
    def get_actor(self):
        """
        Get the sensor object.

        Returns:
            The sensor object.
        """
        return self.sensor_actor

    def listen(self, callback):
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
