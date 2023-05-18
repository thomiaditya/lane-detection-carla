import os
import requests
import zipfile
import sys
import subprocess
from alive_progress import alive_bar

def _extract_with_logging(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        total_files = len(file_list)

        with alive_bar(total_files, title="Extracting", bar=None, spinner="dots_waves", stats=" {rate}") as bar:
            for i, file in enumerate(file_list, start=1):
                zip_ref.extract(file, extract_path)
                bar()
                print(f"Extracted {file}")

def _download_carla_package(carla_version, platform):
    filename = f"CARLA_{carla_version}_{platform}.zip"

    # If file exists, skip download
    if os.path.exists(filename):
        print("CARLA package already downloaded.")
        return filename
    
    print("Downloading CARLA package...")
    package_url = f"https://carla-releases.s3.eu-west-3.amazonaws.com/{platform}/CARLA_{carla_version}.zip"
    
    with requests.Session() as session:
        response = session.get(package_url, stream=True)
        response.raise_for_status()

        temp_file = f"carla.download"
        total_size = int(response.headers.get("content-length", 0))
        chunk_size = 8192

        with open(temp_file, "wb") as f:
            with alive_bar(total_size // chunk_size, title="Downloading") as bar:
                for chunk in response.iter_content(chunk_size):
                    f.write(chunk)
                    bar()
    
    # Change file name to indicate it has been downloaded
    os.rename(temp_file, filename)
    print("CARLA package downloaded successfully.")
    return temp_file

def _extract_carla_package(temp_file):
    print("Extracting CARLA package...")
    _extract_with_logging(temp_file, "lib")
    print("CARLA package extracted successfully.")

def _remove_temp_file(temp_file):
    print("Removing temporary file...")
    os.remove(temp_file)
    print("Temporary file removed.")

def _install_python_dependencies(carla_version):
    print("Installing Python dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pygame", "numpy"])
    print("Python dependencies installed successfully.")

    print("Installing CARLA client library...")
    carla_client_lib_folder = f"lib/CARLA-{carla_version}/PythonAPI/carla/dist"
    carla_client_lib = os.path.join(carla_client_lib_folder, next(f for f in os.listdir(carla_client_lib_folder) if f.endswith('.whl')))
    subprocess.check_call([sys.executable, "-m", "pip", "install", carla_client_lib])
    print("CARLA client library installed successfully.")

def execute(carla_version):

    # Check if CARLA setup has already been completed
    if os.path.exists(f"lib/CARLA-{carla_version}/INSTALL"):
        _install_python_dependencies(carla_version)
        print("CARLA setup has already been completed.")
        return

    print("Starting CARLA setup...")
    platform = "Linux" if sys.platform.startswith("linux") else "Windows"
    temp_file = _download_carla_package(carla_version, platform)
    _extract_carla_package(temp_file)
    _remove_temp_file(temp_file)
    _install_python_dependencies(carla_version)

    # Change name of extracted folder to "CARLA-{version}"
    os.rename(f"lib/WindowsNoEditor", f"lib/CARLA-{carla_version}")

    # Create indicator file to indicate that CARLA setup has been completed
    with open(f"lib/CARLA-{carla_version}/INSTALL", "w") as f:
        f.write(f"CARLA_VERSION={carla_version}\n")

    print("CARLA setup completed successfully.")