import fire
import os
import subprocess

CARLA_VERSION = "0.9.12"

class Simulation(object):
    """
    A class that represents all commands to run the simulation.
    """
    def test(self):
        from .app import Application
        app = Application()
        app.test_run()

class Pipeline(object):
    CARLA_ROOT_PATH = os.path.join("lib", f"CARLA-{CARLA_VERSION}")

    def __init__(self):
        self.simulation = Simulation()

    def install_carla(self):
        """
        Downloads and installs the specified version of CARLA simulator. If the version is not specified, it installs the default version.

        Args:
            carla_version (str, optional): The version of CARLA simulator to install. Defaults to CARLA_VERSION.
        """
        from .utils.download_install_carla import execute
        execute(CARLA_VERSION)

    def connection_test(self, host="localhost", port=2000, timeout=5):
        """
        Tests the connection to the CARLA simulator.

        Args:
            host (str, optional): The host of the CARLA simulator. Defaults to "localhost".
            port (int, optional): The port of the CARLA simulator. Defaults to 2000.
            timeout (int, optional): The timeout in seconds. Defaults to 5.
        """
        from .simulator.SimulatorManager import SimulatorManager
        simulator = SimulatorManager(host, port)
        # Test connection
        if simulator.world is None:
            print("Connection failed.")
        else:
            print(f"Connection successful on {host}:{port}. CARLA version: {simulator.client.get_server_version()}")

    def run_carla_server(self, world_port=2000):
        """
        Starts the CARLA server.

        Args:
            world_port (int, optional): The world port. Defaults to 2000.
        """
        server_executable = os.path.join(self.CARLA_ROOT_PATH, "CarlaUE4.exe")
        if not os.path.exists(server_executable):
            print(f"Could not find CarlaUE4.exe at {server_executable}. Please check your CARLA installation.")
            return

        command = [server_executable, f"-dx11 -windowed -fps=24 -world-port={world_port} -carla-server -benchmark"]
        subprocess.Popen(command)
    
def main():
    """
    The main entry point of the project. It creates an instance of the Pipeline class and allows command line arguments to control its behavior.
    """
    fire.Fire(Pipeline)
