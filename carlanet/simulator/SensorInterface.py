import carla

class SensorInterface:
    """
    Interface for sensors class to implement.

    This interface is used to make sure that all sensors have the same interface (methods and attributes).
    """
    
    def __init__(self, world: carla.World, blueprint_library: carla.BlueprintLibrary, vehicle: carla.Vehicle):
        """
        Initializes the sensor.

        Parameters:
            world (carla.World): The CARLA world object.
            blueprint_library (carla.BlueprintLibrary): The blueprint library.
            vehicle (carla.Vehicle): The vehicle to attach the sensor to.
        """
        raise NotImplementedError
    
    def setup_sensor(self, transform: carla.Transform):
        """
        Set up the sensor with a location and rotation.

        Parameters:
            transform (carla.Transform): The transform to place the sensor.
        """
        raise NotImplementedError

    def process_data(carla_data):
        """
        Process the raw data from the sensor. (static method)

        Parameters:
            raw_data (carla.Image): The raw data from the sensor.

        Returns:
            Processed data.
        """
        raise NotImplementedError
    
    def listen(self, callback: callable):
        """
        Set a callback function to be called every time the sensor receives data.

        Parameters:
            callback (function): The callback function. Should accept one argument - the data from the sensor.
        """
        raise NotImplementedError
    
    def get_sensor(self):
        """
        Get the sensor object.

        Returns:
            The sensor object.
        """
        raise NotImplementedError

    def destroy(self):
        """
        Clean up the sensor, stop it, and detach it from the vehicle.
        """
        raise NotImplementedError
