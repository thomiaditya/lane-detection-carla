import carla
import random
from .SensorInterface import SensorInterface

class SimulatorManager:
    """Class to interact with the CARLA simulator."""

    def __init__(self, host='localhost', port=2000):
        """
        Initializes connection with CARLA simulator.

        Args:
            host (str): The IP address of the machine running the CARLA simulator. Defaults to 'localhost'.
            port (int): The port to use for the connection. Defaults to 2000.
        """
        # Attributes
        self.client = None
        self.world = None
        self.actors = []
        self.blueprint_library = None

        # Connect to CARLA simulator
        self.client = carla.Client(host, port)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()

        # Check if CARLA client is connected to server
        if self.world is None:
            print("Connection failed.")
            return
        print(f"Connection successful on {host}:{port}. CARLA version: {self.client.get_server_version()}")

        self.blueprint_library = self.world.get_blueprint_library()

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
            print(f"Destroyed actor {actor.type_id} with id {actor.id}.")
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
    
    def add_actor(self, actor:carla.Actor):
        """
        Adds an actor to the list of actors.

        Args:
            actor (carla.Actor): The actor to add.
        """
        self.actors.append(actor)