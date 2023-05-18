import fire

CARLA_VERSION = "0.9.14"

class Pipeline(object):
    def install_carla(self, carla_version=CARLA_VERSION):
        """
        Downloads and installs the specified version of CARLA simulator. If the version is not specified, it installs the default version.

        Args:
            carla_version (str, optional): The version of CARLA simulator to install. Defaults to CARLA_VERSION.
        """
        from .utils.download_install_carla import execute
        execute(carla_version)

def main():
    """
    The main entry point of the project. It creates an instance of the Pipeline class and allows command line arguments to control its behavior.
    """
    fire.Fire(Pipeline)
