import fire
import os
import subprocess

CARLA_VERSION = "0.9.12"

class Simulation(object):
    """
    All commands to run the CARLA client simulation.
    """
    def test(self):
        from .app import Application
        app = Application()
        app.test_run()

    def run(self):
        from .app import Application
        app = Application()
        app.run()

class Server(object):
    """
    All commands to run the CARLA server, scenario runner, etc.
    """

    CARLA_ROOT_PATH = os.path.join("lib", f"CARLA-{CARLA_VERSION}")

    def run(self, world_port: int = 2000, low_quality: bool = False):
        """
        Starts the CARLA server.

        Args:
            world_port (int, optional): The world port. Defaults to 2000.
        """
        current_file_path = os.path.dirname(os.path.realpath(__file__))
    
        # Get project root path from current file path. Keep going up one directory until we find the "setup.py" file.
        while not os.path.exists(os.path.join(current_file_path, "setup.py")):
            current_file_path = os.path.dirname(current_file_path)
        
        # Change current working directory to project root path
        os.chdir(current_file_path)

        server_executable = os.path.join(self.CARLA_ROOT_PATH, "CarlaUE4.exe")
        if not os.path.exists(server_executable):
            print(f"Could not find CarlaUE4.exe at {server_executable}. Please check your CARLA installation.")
            return

        command = [server_executable, "-dx11", "-carla-port={}".format(world_port), "-windowed", f"-quality-level={'Low' if low_quality else 'Epic'}", "-benchmark", "-fps=30"]
        subprocess.Popen(command)
        print(f"CARLA server started on port {world_port} with quality level {'Low' if low_quality else 'Epic'}. CARLA version: {CARLA_VERSION}")

class Pipeline(object):

    def __init__(self):
        self.simulation = Simulation()
        self.server = Server()

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
    
def main():
    """
    The main entry point of the project. It creates an instance of the Pipeline class and allows command line arguments to control its behavior.
    """
    fire.Fire(Pipeline)
