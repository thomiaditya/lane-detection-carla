import carla
import random

class SimulatorManager:
    """Class to interact with the CARLA simulator."""

    def __init__(self, host='localhost', port=2000):
        """
        Initializes connection with CARLA simulator.

        Args:
            host (str): The IP address of the machine running the CARLA simulator. Defaults to 'localhost'.
            port (int): The port to use for the connection. Defaults to 2000.
        """
        self.client = carla.Client(host, port)
        self.client.set_timeout(5.0)

        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.actors = []

    def spawn_vehicle(self, vehicle_type='model3') -> carla.Actor:
        """
        Spawns a vehicle in the simulation.

        Args:
            vehicle_type (str): Type of vehicle to spawn. Defaults to 'model3'.

        Returns:
            carla.Actor: The spawned vehicle actor.
        """
        vehicle_bp = self.blueprint_library.filter(vehicle_type)[0]
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.actors.append(vehicle)
        return vehicle

    def spawn_sensor(self, sensor_type, sensor_callback, parent_actor, sensor_transform=None) -> carla.Actor:
        """
        Spawns a sensor and attaches it to a parent actor.

        Args:
            sensor_type (str): Type of sensor to spawn.
            sensor_callback (function): Function to call when the sensor captures data.
            parent_actor (carla.Actor): Actor to attach the sensor to.

        Returns:
            carla.Actor: The spawned sensor actor.
        """
        sensor_bp = self.blueprint_library.filter(sensor_type)[0]
        if sensor_transform is None:
            sensor_location = carla.Location(x=1.5, z=2.4)
            sensor_rotation = carla.Rotation(pitch=-15)
            sensor_transform = carla.Transform(sensor_location, sensor_rotation)
        sensor = self.world.spawn_actor(sensor_bp, sensor_transform, attach_to=parent_actor)
        sensor.listen(sensor_callback)
        self.actors.append(sensor)
        return sensor

    def apply_control(self, actor, throttle=0.0, steer=0.0, brake=0.0, reverse=False, hand_brake=False, manual_gear_shift=False, gear=1):
        """
        Applies control to an actor.

        Args:
            actor (carla.Actor): The actor to apply control to.
            throttle (float): Throttle value to apply. Defaults to 0.0.
            steer (float): Steer value to apply. Defaults to 0.0.
            brake (float): Brake value to apply. Defaults to 0.0.
            reverse (bool): Whether to move in reverse. Defaults to False.
            hand_brake (bool): Whether to apply the hand brake. Defaults to False.
            manual_gear_shift (bool): Whether to enable manual gear shift. Defaults to False.
            gear (int): Gear to set, if manual gear shift is enabled. Defaults to 1.
        """
        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake, reverse=reverse, hand_brake=hand_brake, manual_gear_shift=manual_gear_shift, gear=gear)
        actor.apply_control(control)

    def move_spectator(self, actor):
        """
        Moves the spectator to an actor. The spectator is the camera view in the simulator. Minus z value is used to move the camera above the actor.

        Args:
            actor (carla.Actor): The actor to move the spectator to.
        """
        spectator = self.world.get_spectator()
        spectator.set_transform(carla.Transform(actor.get_location() + carla.Location(z=50, y=-20), carla.Rotation(pitch=-90)))

    def destroy(self):
        """Destroys all spawned actors."""
        for actor in self.actors:
            actor.destroy()
        self.actors = []

    def get_blueprint_library(self) -> carla.BlueprintLibrary:
        """
        Returns the blueprint library.

        Returns:
            carla.BlueprintLibrary: The blueprint library.
        """
        return self.blueprint_library

    def get_world(self) -> carla.World:
        """
        Returns the world.

        Returns:
            carla.World: The world.
        """
        return self.world