import carla
import random
import matplotlib.pyplot as plt
import numpy as np
from .sensors.SensorInterface import SensorInterface

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
        self.sync_mode = False
        self.traffic_manager = None

        # Connect to CARLA simulator
        self.client = carla.Client(host, port)
        print(f"Connecting to {host}:{port}...")
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        # Check if CARLA client is connected to server
        if self.world is None:
            print("Connection failed.")
            return
        print(f"Connection successful on {host}:{port}. CARLA version: {self.client.get_server_version()}")

        self.blueprint_library = self.world.get_blueprint_library()

        # Set the traffic manager
        # TODO: Implement traffic manager in CARLA for testing the model on traffic
        self.traffic_manager = self.client.get_trafficmanager()
        self.traffic_manager.set_global_distance_to_leading_vehicle(1.0)
    
    def tick(self):
        """Tick the CARLA simulator."""
        self.world.tick()
    
    def load_map(self, map_name:str):
        """
        Loads a map in the CARLA simulator.

        Args:
            map_name (str): The name of the map to load.
        """
        self.world = self.client.load_world(map_name)
    
    def set_sync_mode(self, sync_mode:bool):
        """
        Sets the sync mode of the CARLA simulator.

        Args:
            sync_mode (bool): Whether to use synchronous mode or not.
        """
        settings = self.world.get_settings()
        settings.synchronous_mode = sync_mode
        settings.fixed_delta_seconds = 0.05 if sync_mode else None
        self.world.apply_settings(settings)
        self.sync_mode = sync_mode

    def check_sync_mode(self):
        """
        Checks if the CARLA simulator is in synchronous mode.

        Returns:
            bool: Whether the CARLA simulator is in synchronous mode or not.
        """
        return self.sync_mode

    def generate_example_waypoints(self, start_location, distance_between_waypoints, num_waypoints):
        """
        Generates a list of waypoints for a vehicle to follow.

        Parameters:
            start_location (carla.Location): The location to start generating waypoints from.
            distance_between_waypoints (float): The desired distance between waypoints, in meters.
            num_waypoints (int): The number of waypoints to generate.

        Returns:
            A list of waypoints.
        """
        # Get the map from the CARLA world
        carla_map = self.get_world().get_map()

        # Get the closest waypoint to the start location
        start_waypoint = carla_map.get_waypoint(start_location)

        # Initialize the list of waypoints with the start waypoint
        waypoints = [start_waypoint]

        # Generate the remaining waypoints
        for i in range(num_waypoints - 1):

            # Get the next waypoint
            next_waypoints = start_waypoint.next(distance_between_waypoints)

            # If there are no more waypoints, break the loop
            if not next_waypoints:
                break

            # Otherwise, choose the first waypoint from the list of next waypoints
            # Note: The 'next' function can return multiple waypoints if the current waypoint is at an intersection
            next_waypoint = next_waypoints[0]

            # Add the waypoint to the list
            waypoints.append(next_waypoint)

            # Move on to the next waypoint
            start_waypoint = next_waypoint

        return waypoints

    def move_spectator(self, actor):
        """
        Moves the spectator to an actor. The spectator is the camera view in the simulator. Minus z value is used to move the camera above the actor.

        Args:
            actor (carla.Actor): The actor to move the spectator to.
        """
        spectator = self.world.get_spectator()
        
        # Get the actor's location
        actor_location = actor.get_location()

        # Move the spectator behind the actor
        # Get the actor's transform
        actor_transform = actor.get_transform()

        # Calculate the spectator's new location. The spectator will be moved to a position 10 meters behind the actor and 5 meters above.
        spectator_transform = carla.Transform(
            carla.Location(
                x=actor_transform.location.x - 15 * np.cos(actor_transform.rotation.yaw * np.pi / 180.0),
                y=actor_transform.location.y - 15 * np.sin(actor_transform.rotation.yaw * np.pi / 180.0),
                z=actor_transform.location.z + 5
            ),
            actor_transform.rotation
        )

        # Set the spectator's transform
        spectator.set_transform(spectator_transform)

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

    def visualize_control(self, control: carla.VehicleControl):
        """
        Visualize the control commands using plot.

        Parameters:
            control (carla.VehicleControl): The control command for the vehicle.
        """
        # Live plot the control commands
        plt.ion()
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.title('Throttle')
        plt.plot(control.throttle, 'r.')
        plt.subplot(2, 1, 2)
        plt.title('Steering')
        plt.plot(control.steer, 'b.')
        plt.pause(0.001)