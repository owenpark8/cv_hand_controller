from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    model_path = PathJoinSubstitution([
        FindPackageShare("cv_hand_controller"),
        "model",
        "hand_landmarker.task",
    ])

    calibration_path = PathJoinSubstitution([
        FindPackageShare("cv_hand_controller"),
        "calibration",
        "stereo_calibration.yaml",
    ])

    return LaunchDescription([
        Node(
            package="cv_hand_controller",
            executable="camera_node.py",
            name="camera_0",
            output="screen",
            parameters=[{
                "model_path": model_path,
                "camera_index": 0,
                "camera_fps": 30.0,
                "detection_period_ms": 100.0,
                # "output_dir": "/tmp/test/side",
            }],
        ),
        Node(
            package="cv_hand_controller",
            executable="camera_node.py",
            name="camera_1",
            output="screen",
            parameters=[{
                "model_path": model_path,
                "camera_index": 2,
                "camera_fps": 30.0,
                "detection_period_ms": 100.0,
                # "output_dir": "/tmp/test/top",
            }],
        ),
        Node(
            package="cv_hand_controller",
            executable="triangulation_node.py",
            name="triangulation",
            output="screen",
            parameters=[{
                "calibration_yaml_path": calibration_path,
            }],
        ),
        Node(
            package="cv_hand_controller",
            executable="ik_bridge_node.py",
            name="ik_bridge",
            output="screen",
            parameters=[{
                "output_topic": "/ee_pos_cmd",
                "src_map_x": "z",
                "src_map_y": "x",
                "src_map_z": "y",
                "arm_offset_x": 1.25,
                "arm_offset_y": 0.20,
                "arm_offset_z": 0.00,
                "arm_scale_x": -2.0,
                "arm_scale_y": -1.5,
                "arm_scale_z": 2.0,
            }],
        ),
    ])
