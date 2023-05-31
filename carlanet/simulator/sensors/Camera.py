import carla
import numpy as np
from .SensorInterface import SensorInterface

class Camera(SensorInterface):
    """
    Camera sensor class implementing SensorInterface.

    This camera sensor can be attached to a vehicle to capture images from the CARLA simulator.
    """

    def __init__(self, world: carla.World, blueprint_library: carla.BlueprintLibrary, vehicle: carla.Vehicle):
        """
        Initialize the camera sensor.

        The sensor is not set up yet after initialization. Call setup_sensor to set it up.
        """
        # Attributes
        self.blueprint = None
        self.world = world
        self.vehicle = vehicle
        self.sensor_actor = None

        # Set sensor attributes
        self.blueprint = blueprint_library.find('sensor.camera.rgb')
        self.blueprint.set_attribute('image_size_x', '1280')
        self.blueprint.set_attribute('image_size_y', '720')
        self.blueprint.set_attribute('fov', '110')
        
    def set_blueprint_attribute(self, attribute: str, value: str):
        """
        Set an attribute of the blueprint.

        Args:
            attribute (str): The name of the attribute.
            value (str): The value to set the attribute to.
        """
        self.blueprint.set_attribute(attribute, value)

    def setup_sensor(self, transform: carla.Transform):
        """
        Set up the sensor at a specific location and rotation.
        """

        # Set up the sensor
        self.sensor_actor = self.world.spawn_actor(self.blueprint, transform, attach_to=self.vehicle)

    def process_data(carla_data: carla.Image) -> np.ndarray:
        """
        Process the raw sensor data.

        Args:
            carla_data (carla.Image): The raw sensor data from CARLA.

        Returns:
            np.ndarray: The processed data.
        """
        image = carla_data

        # Convert the image to a numpy array
        image.convert(carla.ColorConverter.Raw)
        image_data = image.raw_data
        image_size = (image.height, image.width, 4)
        frame = np.frombuffer(image_data, dtype=np.uint8).reshape(image_size)
        frame = frame[:, :, :3]

        return frame
    
    def listen(self, callback: callable):
        """
        Listen to the sensor and call the callback function when data is captured.

        Args:
            callback (function): The function to call when data is captured.
        """
        return self.sensor_actor.listen(callback)
    
    def get_actor(self) -> carla.Actor:
        """
        Get the sensor object.

        Returns:
            carla.Actor: The sensor object.
        """
        return self.sensor_actor

    def destroy(self):
        """
        Clean up the sensor. Stop it and detach it from the vehicle.
        """
        self.sensor_actor.destroy()
        self.sensor_actor = None
