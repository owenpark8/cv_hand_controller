#!/usr/bin/env python3

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import cv2
import numpy as np
import rclpy
import yaml
from numpy.typing import NDArray
from rclpy.node import Node

from cv_hand_controller.msg import FingerPoints
from message_filters import ApproximateTimeSynchronizer, Subscriber as MFSubscriber

Mat = NDArray[np.float64]


@dataclass
class Intrinsics:
    image_width: int
    image_height: int
    K: Mat
    dist: Mat
    raw: Dict[str, Any]


@dataclass
class Extrinsics:
    camera_index: int
    detected_id: int
    R_world_to_cam: Mat
    t_world_to_cam: Mat
    R_cam_to_world: Mat
    t_cam_to_world: Mat
    P_world_to_cam: Mat


@dataclass
class CameraModel:
    name: str
    intr: Intrinsics
    extr: Extrinsics


class StereoTriangulationNode(Node):
    def __init__(self) -> None:
        super().__init__("stereo_triangulation")

        self.declare_parameter("calibration_yaml_path", "")
        self.declare_parameter("camera_0_topic", "/camera_0/finger_points")
        self.declare_parameter("camera_1_topic", "/camera_1/finger_points")
        self.declare_parameter("sync_queue_size", 10)
        self.declare_parameter("sync_slop_sec", 0.03)

        calib_path_param = self.get_parameter("calibration_yaml_path").get_parameter_value().string_value
        topic0 = self.get_parameter("camera_0_topic").get_parameter_value().string_value
        topic1 = self.get_parameter("camera_1_topic").get_parameter_value().string_value
        queue_size = int(self.get_parameter("sync_queue_size").value)
        slop_sec = float(self.get_parameter("sync_slop_sec").value)

        if not calib_path_param:
            raise RuntimeError("Parameter 'calibration_yaml_path' must be set to a YAML file path")

        calib_path = Path(calib_path_param)
        if not calib_path.is_file():
            raise RuntimeError(f"Calibration YAML not found: {calib_path}")

        self.get_logger().info(f"Loading stereo calibration from: {calib_path}")

        (
            self.m_world_frame,
            self.m_tag_size_m,
            self.m_aruco_dict,
            self.m_target_id,
            self.m_cam0,
            self.m_cam1,
        ) = self._load_calibration_yaml(calib_path)
        self._log_calibration_summary()

        # publisher
        self.m_triangulated_pub = self.create_publisher(
            FingerPoints,
            "/triangulated_finger_points",
            10,
        )

        # subscribers
        self.get_logger().info(
            f"Subscribing to:\n"
            f"  camera_0_topic = {topic0}\n"
            f"  camera_1_topic = {topic1}\n"
            f"  sync_queue_size = {queue_size}, sync_slop_sec = {slop_sec}"
        )
        self.m_sub0 = MFSubscriber(self, FingerPoints, topic0)
        self.m_sub1 = MFSubscriber(self, FingerPoints, topic1)

        self.m_sync_sub = ApproximateTimeSynchronizer(
            [self.m_sub0, self.m_sub1],
            queue_size=queue_size,
            slop=slop_sec,
        )
        self.m_sync_sub.registerCallback(self._sync_callback)

    def _load_calibration_yaml(
        self, path: Path
    ) -> tuple[str, float, int, int, CameraModel, CameraModel]:
        data = yaml.safe_load(path.read_text())

        world_frame: str = data["world_frame"]
        tag_size_m: float = float(data["tag_size_m"])
        aruco_dict: int = int(data["aruco_dictionary"])
        target_id: int = int(data["target_id"])

        cam0_block = data["camera_0"]
        cam1_block = data["camera_1"]

        cam0 = self._parse_camera_block("camera_0", cam0_block)
        cam1 = self._parse_camera_block("camera_1", cam1_block)

        return world_frame, tag_size_m, aruco_dict, target_id, cam0, cam1

    def _parse_camera_block(self, name: str, block: Dict[str, Any]) -> CameraModel:
        intr_block = block["intrinsics"]
        extr_block = block["extrinsics"]

        w = int(intr_block["image_width"])
        h = int(intr_block["image_height"])
        K = np.array(intr_block["camera_matrix"], dtype=np.float64)
        dist = np.array(intr_block["distortion_coefficients"], dtype=np.float64).reshape(-1)

        intr = Intrinsics(image_width=w, image_height=h, K=K, dist=dist, raw=intr_block)

        cam_index = int(extr_block["camera_index"])
        detected_id = int(extr_block["detected_id"])
        R_wc = np.array(extr_block["R_world_to_cam"], dtype=np.float64)
        t_wc = np.array(extr_block["t_world_to_cam"], dtype=np.float64).reshape(3, 1)
        R_cw = np.array(extr_block["R_cam_to_world"], dtype=np.float64)
        t_cw = np.array(extr_block["t_cam_to_world"], dtype=np.float64).reshape(3, 1)
        P_wc = np.array(extr_block["P_world_to_cam"], dtype=np.float64)

        extr = Extrinsics(
            camera_index=cam_index,
            detected_id=detected_id,
            R_world_to_cam=R_wc,
            t_world_to_cam=t_wc,
            R_cam_to_world=R_cw,
            t_cam_to_world=t_cw,
            P_world_to_cam=P_wc,
        )
        return CameraModel(name=name, intr=intr, extr=extr)

    def _sync_callback(self, msg0: FingerPoints, msg1: FingerPoints) -> None:
        thumb0 = list(msg0.thumb_points)
        thumb1 = list(msg1.thumb_points)
        index0 = list(msg0.index_points)
        index1 = list(msg1.index_points)

        all_inputs = thumb0 + thumb1 + index0 + index1
        if any(math.isnan(v) for v in all_inputs):
            # tracking lost on at least one finger
            return

        thumb_world = self._triangulate_pair(thumb0, thumb1)
        if thumb_world is None:
            return

        index_world = self._triangulate_pair(index0, index1)
        if index_world is None:
            return

        t0 = msg0.header.stamp
        t1 = msg1.header.stamp
        avg_sec = (t0.sec + t1.sec) // 2
        avg_nanosec = (t0.nanosec + t1.nanosec) // 2

        out_msg = FingerPoints()
        out_msg.header.stamp.sec = avg_sec
        out_msg.header.stamp.nanosec = avg_nanosec
        out_msg.header.frame_id = self.m_world_frame

        out_msg.thumb_points = thumb_world
        out_msg.index_points = index_world

        self.m_triangulated_pub.publish(out_msg)

    def _triangulate_pair(self, pt0: List[float], pt1: List[float]) -> Optional[List[float]]:
        x0_norm, y0_norm, _ = pt0
        x1_norm, y1_norm, _ = pt1

        # convert normalized [0,1] coords to pixel coordinates
        u0 = x0_norm * self.m_cam0.intr.image_width
        v0 = y0_norm * self.m_cam0.intr.image_height

        u1 = x1_norm * self.m_cam1.intr.image_width
        v1 = y1_norm * self.m_cam1.intr.image_height

        # triangulate
        Xw = self._triangulate_world(u0, v0, u1, v1)

        if Xw is None:
            return None

        return [float(Xw[0]), float(Xw[1]), float(Xw[2])]

    def _triangulate_world(self, u0: float, v0: float, u1: float, v1: float) -> Optional[np.ndarray]:
        # undistort & normalize pixel to camera coordinates
        x0, y0 = self._normalize_pixel(u0, v0, self.m_cam0.intr)
        x1, y1 = self._normalize_pixel(u1, v1, self.m_cam1.intr)

        # build projection matrices using only extrinsics [R|t]
        P0 = np.hstack((self.m_cam0.extr.R_world_to_cam, self.m_cam0.extr.t_world_to_cam))  # 3x4
        P1 = np.hstack((self.m_cam1.extr.R_world_to_cam, self.m_cam1.extr.t_world_to_cam))  # 3x4

        # triangulate
        pts0 = np.array([[x0], [y0]], dtype=np.float64)
        pts1 = np.array([[x1], [y1]], dtype=np.float64)

        X_h = cv2.triangulatePoints(P0, P1, pts0, pts1)  # 4x1
        w = X_h[3, 0]
        if abs(w) < 1e-9:
            return None

        X = X_h[:3, 0] / w  # 3D in world frame
        return X

    @staticmethod
    def _normalize_pixel(u: float, v: float, intr: Intrinsics) -> Tuple[float, float]:
        pts = np.array([[[u, v]]], dtype=np.float64)
        norm = cv2.undistortPoints(pts, intr.K, intr.dist, P=None)
        x, y = norm[0, 0]
        return float(x), float(y)

    def _log_calibration_summary(self) -> None:
        self.get_logger().info(f"World frame      : {self.m_world_frame}")
        self.get_logger().info(f"Tag size [m]     : {self.m_tag_size_m:.5f}")
        self.get_logger().info(f"ArUco dict       : {self.m_aruco_dict}")
        self.get_logger().info(f"Target ID        : {self.m_target_id}")
        self.get_logger().info("")
        for cam in (self.m_cam0, self.m_cam1):
            self._log_camera_summary(cam)

    def _log_camera_summary(self, cam: CameraModel) -> None:
        intr = cam.intr
        extr = cam.extr
        self.get_logger().info(f"[{cam.name}] camera_index = {extr.camera_index}")
        self.get_logger().info(f"[{cam.name}] image size  = {intr.image_width} x {intr.image_height}")
        fx = intr.K[0, 0]; fy = intr.K[1, 1]; cx = intr.K[0, 2]; cy = intr.K[1, 2]
        self.get_logger().info(f"[{cam.name}] fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        pos = extr.t_cam_to_world.flatten()
        self.get_logger().info(f"[{cam.name}] cam pos (world) = [{pos[0]: .3f}, {pos[1]: .3f}, {pos[2]: .3f}] m")
        yaw, pitch, roll = self._rotation_to_ypr(extr.R_cam_to_world)
        self.get_logger().info(f"[{cam.name}] cam ypr (deg) = [{yaw: .1f}, {pitch: .1f}, {roll: .1f}]\n")

    @staticmethod
    def _rotation_to_ypr(R: Mat) -> tuple[float, float, float]:
        sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            yaw = math.atan2(R[1, 0], R[0, 0])
            pitch = math.atan2(-R[2, 0], sy)
            roll = math.atan2(R[2, 1], R[2, 2])
        else:
            yaw = math.atan2(-R[0, 1], R[1, 1])
            pitch = math.atan2(-R[2, 0], sy)
            roll = 0.0
        return (math.degrees(yaw), math.degrees(pitch), math.degrees(roll))


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = StereoTriangulationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
