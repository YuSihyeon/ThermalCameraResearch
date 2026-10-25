from glob import glob
from setuptools import find_packages, setup


package_name = "firesight_calib"

setup(
    name=package_name,
    version="0.3.0",
    packages=find_packages(exclude=("test",)),
    package_data={"firesight_calib.experiments": ["profiles/*.yaml"]},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/docs", glob("docs/*.md")),
    ],
    # ROS installs these through rosdep/apt to preserve Humble's NumPy/OpenCV ABI.
    # Standalone pip installs use the repository-root pyproject.toml instead.
    install_requires=["setuptools", "numpy", "PyYAML"],
    python_requires=">=3.10",
    zip_safe=True,
    maintainer="HaechiAR",
    maintainer_email="dev@haechiar.local",
    description="FireSight sensor-rig calibration tools",
    license="Proprietary",
    entry_points={
        "console_scripts": [
            "firesight-calib = firesight_calib.cli:main",
            "calib_home = firesight_calib.cli:home",
        ],
    },
)
