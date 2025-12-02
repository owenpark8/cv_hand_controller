#!/usr/bin/env python3

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import rclpy
import yaml
from numpy.typing import NDArray
from rclpy.node import Node


Mat = NDArray[np.float64]


@dataclass
class Intrinsics:
    image_width: int
    image_height: int
    K: Mat              # 3x3
    dist: Mat           # (N,) or (N,1)
    raw: Dict[str, Any]


@dataclass
class Extrinsics:
    camera_index: int
    detected_id: int
    R_world_to_cam: Mat   # 3x3
    t_world_to_cam: Mat   # (3,1)
    R_cam_to_world: Mat   # 3x3
    t_cam_to_world: Mat   # (3,1)
    P_world_to_cam: Mat   # 3x4


@dataclass
class CameraModel:
    name: str
    intr: Intrinsics
    extr: Extrinsics


class StereoTriangulationNode(Node):
    def __init__(self) -> None:
        super().__init__("stereo_triangulation")

        self.declare_parameter("calibration_yaml_path", "")
        calib_path_param = self.get_parameter("calibration_yaml_path").get_parameter_value().string_value

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

        # TODO: in next step:
        #  - subscribe to camera_0 and camera_1 FingerPoints topics
        #  - triangulate corresponding points into world frame
        #  - publish geometry_msgs/PointStamped


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

        # intrinsics
        w = int(intr_block["image_width"])
        h = int(intr_block["image_height"])

        K = np.array(intr_block["camera_matrix"], dtype=np.float64)
        dist = np.array(intr_block["distortion_coefficients"], dtype=np.float64).reshape(-1)

        intr = Intrinsics(
            image_width=w,
            image_height=h,
            K=K,
            dist=dist,
            raw=intr_block,
        )

        # extrinsics
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
        self.get_logger().info(
            f"[{cam.name}] image size  = {intr.image_width} x {intr.image_height}"
        )

        fx = intr.K[0, 0]
        fy = intr.K[1, 1]
        cx = intr.K[0, 2]
        cy = intr.K[1, 2]
        self.get_logger().info(
            f"[{cam.name}] fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}"
        )

        # quick position of camera in world frame (t_cam_to_world)
        pos = extr.t_cam_to_world.flatten()
        self.get_logger().info(
            f"[{cam.name}] cam position (world) = "
            f"[{pos[0]: .3f}, {pos[1]: .3f}, {pos[2]: .3f}] m"
        )

        # quick orientation check: yaw/pitch/roll from R_cam_to_world
        yaw, pitch, roll = self._rotation_to_ypr(extr.R_cam_to_world)
        self.get_logger().info(
            f"[{cam.name}] cam yaw/pitch/roll (deg) = "
            f"[{yaw: .1f}, {pitch: .1f}, {roll: .1f}]"
        )
        self.get_logger().info("")

    @staticmethod
    def _rotation_to_ypr(R: Mat) -> tuple[float, float, float]:
        """
        Convert rotation matrix to yaw/pitch/roll (Z-Y-X, in degrees).
        Just for logging sanity; not used in math.
        """
        sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            yaw = math.atan2(R[1, 0], R[0, 0])
            pitch = math.atan2(-R[2, 0], sy)
            roll = math.atan2(R[2, 1], R[2, 2])
        else:
            # Gimbal lock
            yaw = math.atan2(-R[0, 1], R[1, 1])
            pitch = math.atan2(-R[2, 0], sy)
            roll = 0.0

        return (
            math.degrees(yaw),
            math.degrees(pitch),
            math.degrees(roll),
        )


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
