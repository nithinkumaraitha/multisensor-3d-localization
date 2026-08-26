from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'u_drive_perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', [os.path.join('resource', package_name)]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='udrive',
    maintainer_email='you@example.com',
    description='YOLO + LiDAR fusion with GNSS: publishes object names, local (x,y,z), global (lat,lon,alt), and overlay image.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_lidar_fusion_node = u_drive_perception.yolo_lidar_fusion_node:main',
            'yolo_detector_node = u_drive_perception.yolo_detector_node:main',
            'objects_printer_node = u_drive_perception.objects_printer_node:main',
        ],
    },
)

