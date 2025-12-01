import os

from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import rclpy

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from rclpy.node import Node
from rclpy.parameter import Parameter

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


def draw_landmarks_on_image(rgb_image: np.ndarray,
                            detection_result: HandLandmarkerResult) -> np.ndarray:
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z
            ) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style(),
        )

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(
            annotated_image,
            f"{handedness[0].category_name}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            HANDEDNESS_TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return annotated_image

class CvHandController(Node):
    def __init__(self) -> None:
        super().__init__("cv_hand_controller")
        self.get_logger().info("cv_hand_controller node started")

        # parameters
        self.declare_parameter("model_path", "")
        self.declare_parameter("camera_index", 0)  # default: /dev/video0
        self.declare_parameter("camera_fps", 30.0)
        self.declare_parameter("detection_period_ms", 1000.0)
        self.declare_parameter("output_dir", "")

        self.m_model_path: str = self.get_parameter("model_path").value
        self.m_camera_index: int = self.get_parameter("camera_index").value
        self.m_camera_fps: float = self.get_parameter("camera_fps").value
        self.m_detection_period_ms: float = self.get_parameter("detection_period_ms").value
        self.m_output_dir: str = self.get_parameter("output_dir").value

        self.m_save_enabled: bool = bool(self.m_output_dir)
        self.m_frame_counter: int = 0
        self.m_inference_in_flight: bool = False

        if not self.m_model_path:
            raise RuntimeError("Parameter 'model_path' must be set")

        if self.m_camera_fps <= 0.0:
            self.get_logger().warn("camera_fps <= 0, defaulting to 30.0")
            self.m_camera_fps = 30.0

        if self.m_detection_period_ms <= 0.0:
            self.get_logger().warn(
                "detection_period_ms <= 0, defaulting to 1000 ms"
            )
            self.m_detection_period_ms = 1000.0

        if self.m_save_enabled:
            os.makedirs(self.m_output_dir, exist_ok=True)
            self.get_logger().info(f"Saving annotated frames to: {self.m_output_dir}")
        else:
            self.get_logger().info("output_dir not set → frame saving disabled")

        # opencv camera (YUYV 640x480)
        self.m_cap: Optional[cv2.VideoCapture] = cv2.VideoCapture(
            self.m_camera_index, cv2.CAP_V4L2
        )
        if not self.m_cap.isOpened():
            raise RuntimeError(f"Failed to open camera index {self.m_camera_index}")

        self.m_cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc("Y", "U", "Y", "V"),
        )
        self.m_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.m_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.m_cap.set(cv2.CAP_PROP_FPS, self.m_camera_fps)

        # log camera settings
        fourcc_int = int(self.m_cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join(
            chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)
        )
        width = int(self.m_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.m_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.m_cap.get(cv2.CAP_PROP_FPS)

        self.get_logger().info(
            f"Camera {self.m_camera_index}: {width}x{height} "
            f"@ {actual_fps:.1f} fps, FOURCC={fourcc_str}"
        )

        # mediapipe landmarker
        def landmarker_result_callback(
            result: HandLandmarkerResult,
            output_image: mp.Image,
            timestamp_ms: int,
        ) -> None:
            self._on_result(result, output_image, timestamp_ms)

        options: HandLandmarkerOptions = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.m_model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_hands=1,
            result_callback=landmarker_result_callback,
        )
        self.m_landmarker: Optional[HandLandmarker] = (
            HandLandmarker.create_from_options(options)
        )

        period_sec = self.m_detection_period_ms / 1000.0
        self.m_timer = self.create_timer(period_sec, self._maybe_grab_frame)


    def _maybe_grab_frame(self) -> None:
        self.get_logger().info("Grab frame called")

        if self.m_inference_in_flight:
            self.get_logger().warn("Inference in flight; skipping grab")
            return

        if self.m_landmarker is None or self.m_cap is None:
            self.get_logger().warn("Landmarker or camera not ready; skipping grab")
            return

        ret, frame_bgr = self.m_cap.read()
        if not ret:
            self.get_logger().warn("Failed to read frame from camera")
            return

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb,
        )

        timestamp_ms: int = self.get_clock().now().nanoseconds // 1_000_000

        self.m_inference_in_flight = True
        self.m_landmarker.detect_async(mp_image, timestamp_ms)


    def _on_result(
        self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int
    ) -> None:
        has_hands = (
            result is not None
            and result.hand_landmarks is not None
            and len(result.hand_landmarks) > 0
        )

        if has_hands:
            self.get_logger().info("Hand(s) detected")
        else:
            self.get_logger().info("No hands detected")

        if self.m_save_enabled:
            rgb_image = output_image.numpy_view()

            if has_hands:
                annotated_rgb = draw_landmarks_on_image(rgb_image, result)
            else:
                annotated_rgb = rgb_image

            annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

            self.m_frame_counter += 1
            filename = os.path.join(
                self.m_output_dir,
                f"frame_{timestamp_ms}_{self.m_frame_counter:06d}.png",
            )
            ok = cv2.imwrite(filename, annotated_bgr)
            if not ok:
                self.get_logger().warn(f"Failed to save frame to {filename}")
            else:
                self.get_logger().info(
                    f"[{timestamp_ms}] Saved frame: {filename}"
                )

        self.m_inference_in_flight = False


    def destroy_node(self) -> bool:
        if self.m_landmarker is not None:
            self.m_landmarker.close()
            self.m_landmarker = None
        if self.m_cap is not None:
            self.m_cap.release()
            self.m_cap = None
        return super().destroy_node()

def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)

    node: CvHandController = CvHandController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
