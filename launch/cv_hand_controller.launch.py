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
    ])
