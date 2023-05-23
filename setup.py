from setuptools import setup, find_packages

setup(
    name="carlanet",
    version="0.1",
    description="Research project on Applying Deep Learning to Autonomous Driving in CARLA Simulator",
    author="Thomi Aditya Alhakiim",
    author_email="thomiaditya@gmail.com",
    packages=find_packages(),
    url="https://github.com/thomiaditya/carlanet",
    install_requires=[
        "fire",
        "numpy",
        "matplotlib",
        "opencv-python",
        "requests",
        "alive-progress",
    ],
    entry_points={
        "console_scripts": [
            "carlanet=carlanet.app:main"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)