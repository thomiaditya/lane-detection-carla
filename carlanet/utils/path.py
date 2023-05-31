# This file is the utility file for maintaining the path.
import os

def get_project_root() -> str:
    """
    Returns the project root path.

    Returns:
        str: The project root path.
    """
    current_file_path = os.path.dirname(os.path.realpath(__file__))

    # Get project root path from current file path. Keep going up one directory until we find the "setup.py" file.
    while not os.path.exists(os.path.join(current_file_path, "setup.py")):
        current_file_path = os.path.dirname(current_file_path)

    return current_file_path

def get_carla_root() -> str:
    """
    Returns the CARLA root path.

    Returns:
        str: The CARLA root path.
    """
    return os.path.join(get_project_root(), "lib", f"CARLA-{os.environ.get('CARLA_VERSION')}")