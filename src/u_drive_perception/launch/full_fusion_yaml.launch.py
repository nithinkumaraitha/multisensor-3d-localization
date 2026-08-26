from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    default_yaml = os.path.join(
        get_package_share_directory('u_drive_perception'),
        'config', 'fusion.yaml'
    )
    args = [DeclareLaunchArgument('params_file', default_value=default_yaml)]

    detector = Node(
        package='u_drive_perception',
        executable='yolo_detector_node',
        name='yolo_detector',
        output='screen',
        parameters=[LaunchConfiguration('params_file')]
    )
    fusion = Node(
        package='u_drive_perception',
        executable='yolo_lidar_fusion_node',
        name='yolo_lidar_fusion',
        output='screen',
        parameters=[LaunchConfiguration('params_file')]
    )
    return LaunchDescription(args + [detector, fusion])

