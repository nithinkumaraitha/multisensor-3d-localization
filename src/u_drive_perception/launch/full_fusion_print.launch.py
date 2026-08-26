from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    args = [
        # IO topics
        DeclareLaunchArgument('image_topic', default_value='/sensing/camera/camera0/image_rect_color'),
        DeclareLaunchArgument('image_compressed_topic', default_value='/sensing/camera/camera0/image_rect_color/compressed'),
        DeclareLaunchArgument('camera_info_topic', default_value='/sensing/camera/camera0/camera_info'),
        DeclareLaunchArgument('cloud_topic', default_value='/sensing/lidar/top/outlier_filtered/pointcloud'),
        DeclareLaunchArgument('gnss_topic', default_value='/sensing/gnss/nav_sat_fix'),
        DeclareLaunchArgument('yolo_topic', default_value='/perception/yolo_dets'),

        # Outputs
        DeclareLaunchArgument('overlay_topic', default_value='/perception/fusion_image'),
        DeclareLaunchArgument('objects_topic', default_value='/perception/objects'),
        DeclareLaunchArgument('markers_topic', default_value='/perception/markers'),
        DeclareLaunchArgument('status_topic', default_value='/perception/status'),

        # Frames
        DeclareLaunchArgument('camera_frame_id', default_value='camera_front_optical_frame'),
        DeclareLaunchArgument('lidar_frame_id', default_value='velodyne_top_base_link'),
        DeclareLaunchArgument('fixed_frame', default_value='velodyne_top_base_link'),

        # Intrinsics fallback
        DeclareLaunchArgument('fx', default_value='700.0'),
        DeclareLaunchArgument('fy', default_value='700.0'),
        DeclareLaunchArgument('cx', default_value='400.0'),
        DeclareLaunchArgument('cy', default_value='300.0'),
        DeclareLaunchArgument('use_camera_info', default_value='true'),

        # Extrinsics
        DeclareLaunchArgument('use_tf', default_value='false'),
        DeclareLaunchArgument('tf_timeout_sec', default_value='0.20'),
        DeclareLaunchArgument('t_lc_x', default_value='0.0'),
        DeclareLaunchArgument('t_lc_y', default_value='0.0'),
        DeclareLaunchArgument('t_lc_z', default_value='1.0'),
        DeclareLaunchArgument('yaw_deg', default_value='0.0'),
        DeclareLaunchArgument('pitch_deg', default_value='0.0'),
        DeclareLaunchArgument('roll_deg', default_value='0.0'),

        # Detector
        DeclareLaunchArgument('det_model',   default_value='/home/nithin/models/yolov8n.pt'),
        DeclareLaunchArgument('det_img_size', default_value='640'),
        DeclareLaunchArgument('det_min_conf', default_value='0.25'),
        DeclareLaunchArgument('det_iou',      default_value='0.50'),
        DeclareLaunchArgument('det_max_det',  default_value='200'),
        DeclareLaunchArgument('det_frame_skip', default_value='1'),
        DeclareLaunchArgument('det_max_fps',    default_value='18.0'),
        DeclareLaunchArgument('det_min_box_area', default_value='32'),
        DeclareLaunchArgument('det_classes_filter', default_value="['person','car','bus','truck','train','bicycle','motorcycle','traffic light','stop sign']"),
    ]

    detector = Node(
        package='u_drive_perception',
        executable='yolo_detector_node',
        name='yolo_detector',
        output='screen',
        parameters=[{
            'image_topic': LaunchConfiguration('image_topic'),
            'image_compressed_topic': LaunchConfiguration('image_compressed_topic'),
            'det_topic': LaunchConfiguration('yolo_topic'),
            'model': LaunchConfiguration('det_model'),
            'img_size': ParameterValue(LaunchConfiguration('det_img_size'), value_type=int),
            'min_conf': ParameterValue(LaunchConfiguration('det_min_conf'), value_type=float),
            'iou_thres': ParameterValue(LaunchConfiguration('det_iou'), value_type=float),
            'max_det': ParameterValue(LaunchConfiguration('det_max_det'), value_type=int),
            'frame_skip': ParameterValue(LaunchConfiguration('det_frame_skip'), value_type=int),
            'max_fps': ParameterValue(LaunchConfiguration('det_max_fps'), value_type=float),
            'min_box_area': ParameterValue(LaunchConfiguration('det_min_box_area'), value_type=int),
            'classes_filter': LaunchConfiguration('det_classes_filter'),
        }]
    )

    fusion = Node(
        package='u_drive_perception',
        executable='yolo_lidar_fusion_node',
        name='yolo_lidar_fusion',
        output='screen',
        parameters=[{
            # topics
            'image_topic': LaunchConfiguration('image_topic'),
            'image_compressed_topic': LaunchConfiguration('image_compressed_topic'),
            'camera_info_topic': LaunchConfiguration('camera_info_topic'),
            'cloud_topic': LaunchConfiguration('cloud_topic'),
            'gnss_topic': LaunchConfiguration('gnss_topic'),
            'yolo_topic': LaunchConfiguration('yolo_topic'),

            # outputs
            'overlay_topic': LaunchConfiguration('overlay_topic'),
            'objects_topic': LaunchConfiguration('objects_topic'),
            'markers_topic': LaunchConfiguration('markers_topic'),
            'status_topic': LaunchConfiguration('status_topic'),

            # frames
            'camera_frame_id': LaunchConfiguration('camera_frame_id'),
            'lidar_frame_id': LaunchConfiguration('lidar_frame_id'),
            'fixed_frame': LaunchConfiguration('fixed_frame'),

            # intrinsics (typed)
            'fx': ParameterValue(LaunchConfiguration('fx'), value_type=float),
            'fy': ParameterValue(LaunchConfiguration('fy'), value_type=float),
            'cx': ParameterValue(LaunchConfiguration('cx'), value_type=float),
            'cy': ParameterValue(LaunchConfiguration('cy'), value_type=float),
            'use_camera_info': ParameterValue(LaunchConfiguration('use_camera_info'), value_type=bool),

            # extrinsics
            'use_tf': ParameterValue(LaunchConfiguration('use_tf'), value_type=bool),
            'tf_timeout_sec': ParameterValue(LaunchConfiguration('tf_timeout_sec'), value_type=float),
            't_lc_x': ParameterValue(LaunchConfiguration('t_lc_x'), value_type=float),
            't_lc_y': ParameterValue(LaunchConfiguration('t_lc_y'), value_type=float),
            't_lc_z': ParameterValue(LaunchConfiguration('t_lc_z'), value_type=float),
            'yaw_deg': ParameterValue(LaunchConfiguration('yaw_deg'), value_type=float),
            'pitch_deg': ParameterValue(LaunchConfiguration('pitch_deg'), value_type=float),
            'roll_deg': ParameterValue(LaunchConfiguration('roll_deg'), value_type=float),

            # fusion behavior
            'min_conf': ParameterValue(LaunchConfiguration('det_min_conf'), value_type=float),
            'expand_px': 14,
            'depth_band_beta': 0.6,
            'reuse_last_ms': 600,
            'lidar_downsample': 1,
            'show_raw_yolo': True,
            'lidar_max_points': 160000,
            'lidar_msg_skip': 0,
        }]
    )

    printer = Node(
        package='u_drive_perception',
        executable='objects_printer_node',
        name='objects_printer',
        output='screen',
        parameters=[{
            'objects_topic': LaunchConfiguration('objects_topic')
        }]
    )

    return LaunchDescription(args + [detector, fusion, printer])

