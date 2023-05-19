import time
from .simulator.SimulatorManager import SimulatorManager

class Application:
    """
    This class manages the main application flow of a CARLA simulation. It initializes a connection to the 
    CARLA simulator, spawns a vehicle, attaches a camera sensor to the vehicle, runs the simulation for a 
    specified duration, and finally cleans up the simulation.

    The application's run method implements a simple driving behavior, in which the vehicle drives straight 
    at a constant throttle.

    Attributes:
        simulator_manager (SimulatorManager): An instance of SimulatorManager which handles the 
        interactions with the CARLA simulator.
    """
    def __init__(self, host='localhost', port=2000):
        self.simulator_manager = SimulatorManager(host, port)
    
    def run(self):
        """
        Runs the main application flow.
        """
        # Spawn a vehicle
        vehicle = self.simulator_manager.spawn_vehicle()

        # Spawn a camera sensor and attach it to the vehicle
        camera_callback = lambda image: print(f"Camera captured an image at timestamp {image.timestamp}")
        self.simulator_manager.spawn_sensor('sensor.camera.rgb', camera_callback, vehicle)

    def test_run(self):
        """
        Runs the test application flow.
        """
        # Spawn a vehicle
        vehicle = self.simulator_manager.spawn_vehicle()

        # Spawn a camera sensor and attach it to the vehicle
        camera_callback = lambda image: print(f"Camera captured an image at timestamp {image.timestamp}")
        self.simulator_manager.spawn_sensor('sensor.camera.rgb', camera_callback, vehicle)

        # Move the spectator to the vehicle
        self.simulator_manager.move_spectator(vehicle)

        # Run the simulation for a certain duration
        simulation_duration = 10  # seconds
        start_time = time.time()
        while time.time() - start_time < simulation_duration:
            # Apply control to the vehicle (e.g., throttle, steering, etc.)
            self.simulator_manager.apply_control(vehicle, throttle=0.5)
            time.sleep(0.1)

        # Clean up the simulation
        self.simulator_manager.destroy()