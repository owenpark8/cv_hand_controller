import mediapipe as mp
import rclpy
from rclpy.node import Node

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

class CvHandController(Node):
    def __init__(self):
        super().__init__("cv_hand_controller")
        self.get_logger().info("cv_hand_controller node started")

def print_result(result: HandLandmarkerResult):
    print('hand landmarker result: {}'.format(result))

def main():
    node = CvHandController()

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='/home/owen/ros2_ws/src/cv_hand_controller/model/hand_landmarker.task'),
        running_mode=VisionRunningMode.IMAGE)

    with HandLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image.create_from_file('/home/owen/ros2_ws/src/cv_hand_controller/test/image.jpg')
        hand_landmarker_result = landmarker.detect(mp_image)
        print_result(hand_landmarker_result)

    print('Hi from hand_arm_controller.')
    rclpy.spin(node)
