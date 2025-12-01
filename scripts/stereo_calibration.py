#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

Mat = NDArray[np.float64]

def print_aruco_tag(args: argparse.Namespace) -> None:
    aruco_dict = cv2.aruco.getPredefinedDictionary(args.aruco_dict)

    tag_px: int = args.tag_px
    tag_id: int = args.tag_id

    tag_img = cv2.aruco.generateImageMarker(
        dictionary=aruco_dict,
        id=tag_id,
        sidePixels=tag_px,
        borderBits=1,
    )

    out_path = Path(args.output_tag)
    ok = cv2.imwrite(str(out_path), tag_img)
    if not ok:
        raise RuntimeError(f"Failed to write ArUco tag to {out_path}")

    print(
        f"ArUco tag written to {out_path.resolve()}\n"
        f"  tag id:      {tag_id}\n"
        f"  dictionary:  {args.aruco_dict}\n"
        f"  resolution:  {tag_px} x {tag_px} px"
    )

def load_intrinsics(json_path: Path) -> Tuple[Mat, Mat, int, int, Dict[str, Any]]:
    """
    Load camera intrinsics from a JSON file of the form:

    {
      "image_width": ...,
      "image_height": ...,
      "camera_matrix": [[...],[...],[...]],
      "distortion_coefficients": [...],
      ...
    }
    Returns (K, dist, width, height, full_json_dict).
    """
    with json_path.open("r") as f:
        data: Dict[str, Any] = json.load(f)

    width: int = int(data["image_width"])
    height: int = int(data["image_height"])

    K = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["distortion_coefficients"], dtype=np.float64).reshape(-1, 1)

    return K, dist, width, height, data


def capture_extrinsics_for_camera(
    camera_index: int,
    K: Mat,
    dist: Mat,
    img_w: int,
    img_h: int,
    tag_size_m: float,
    aruco_dict_id: int,
    target_id: int | None,
    window_name: str,
) -> Dict[str, Any]:
    """
    Open the given camera, detect an ArUco tag, and compute extrinsics.

    world frame = ArUco tag frame
    X_cam = R_world_to_cam * X_world + t_world_to_cam

    Returns a dict containing:
      - camera_index
      - detected_id
      - R_world_to_cam, t_world_to_cam
      - R_cam_to_world, t_cam_to_world
      - P_world_to_cam (projection matrix K [R|t])
    """
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera index {camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_h)

    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
    det_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, det_params)

    print(
        f"[cam {camera_index}] Looking for ArUco tag "
        f"(size={tag_size_m} m, dict={aruco_dict_id})"
    )
    print("Controls:")
    print("  c = capture frame")
    print("  q = abort this camera")

    R_world_to_cam: Mat | None = None
    t_world_to_cam: Mat | None = None
    detected_id: int | None = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _rejected = detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            if target_id is not None:
                mask = (ids.flatten() == target_id)
                if not np.any(mask):
                    cv2.putText(
                        frame,
                        f"IDs: {ids.flatten().tolist()} (waiting for ID {target_id})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )
                else:
                    sel_idx = np.where(mask)[0][0]
                    corners = [corners[sel_idx]]
                    ids = ids[mask]
            else:
                cv2.putText(
                    frame,
                    f"Detected IDs: {ids.flatten().tolist()}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            rvecs, tvecs, _obj_pts = cv2.aruco.estimatePoseSingleMarkers(
                corners, tag_size_m, K, dist
            )

            rvec = rvecs[0]
            tvec = tvecs[0]
            detected_id = int(ids[0, 0])

            cv2.drawFrameAxes(
                frame, K, dist, rvec, tvec, tag_size_m * 0.5
            )

            cv2.putText(
                frame,
                f"ID {detected_id}: press 'c' to capture",
                (10, img_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            R, _ = cv2.Rodrigues(rvec)
            t = tvec.reshape(3, 1)
            R_world_to_cam = R
            t_world_to_cam = t

        else:
            cv2.putText(
                frame,
                "No ArUco tag detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            if R_world_to_cam is None or t_world_to_cam is None or detected_id is None:
                print("[WARN] No valid pose yet; can't capture.")
                continue
            print(f"[cam {camera_index}] Captured pose for tag ID {detected_id}")
            break
        elif key == ord("q"):
            print(f"[cam {camera_index}] Aborted by user.")
            R_world_to_cam = None
            t_world_to_cam = None
            detected_id = None
            break

    cap.release()
    cv2.destroyWindow(window_name)

    if R_world_to_cam is None or t_world_to_cam is None or detected_id is None:
        raise RuntimeError(f"Failed to capture extrinsics for camera {camera_index}")

    # inverse: camera -> world
    R_cam_to_world = R_world_to_cam.T
    t_cam_to_world = -R_world_to_cam.T @ t_world_to_cam

    # projection matrix P = K [R|t] (maps world -> pixels)
    Rt = np.hstack((R_world_to_cam, t_world_to_cam))  # 3x4
    P_world_to_cam = K @ Rt  # 3x4

    return {
        "camera_index": camera_index,
        "detected_id": detected_id,
        # world = ArUco tag frame
        # X_cam = R_world_to_cam * X_world + t_world_to_cam
        "R_world_to_cam": R_world_to_cam.tolist(),
        "t_world_to_cam": t_world_to_cam.flatten().tolist(),
        # X_world = R_cam_to_world * X_cam + t_cam_to_world
        "R_cam_to_world": R_cam_to_world.tolist(),
        "t_cam_to_world": t_cam_to_world.flatten().tolist(),
        # Projection matrix (for convenience)
        "P_world_to_cam": P_world_to_cam.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute full stereo calibration (intrinsics + extrinsics) for two cameras "
            "relative to an ArUco tag, using existing intrinsics JSON files."
        )
    )

    parser.add_argument("--cam0-json", type=str, help="Intrinsics JSON for camera 0")
    parser.add_argument("--cam1-json", type=str, help="Intrinsics JSON for camera 1")
    parser.add_argument("--cam0-index", type=int, default=0, help="VideoCapture index for camera 0")
    parser.add_argument("--cam1-index", type=int, default=2, help="VideoCapture index for camera 1")

    parser.add_argument("--tag-size-m", type=float, help="Physical side length of the ArUco tag in meters")

    parser.add_argument("--aruco-dict", type=int, default=cv2.aruco.DICT_4X4_50, help="ArUco dictionary id, e.g., cv2.aruco.DICT_4X4_50")
    parser.add_argument("--target-id", type=int, default=-1, help="Specific ArUco ID to use (if >= 0); otherwise accept any")

    parser.add_argument("--output-json", type=str, default="stereo_calibration_aruco_world.json", help="Output JSON file for full stereo calibration")

    parser.add_argument("--print-tag", action="store_true")
    parser.add_argument("--tag-id", type=int, default=0, help="ArUco marker ID to print")
    parser.add_argument("--tag-px", type=int, default=600, help="Printed tag size in pixels")
    parser.add_argument("--output-tag", type=str, default="aruco_tag.png", help="Output PNG path for printed ArUco tag")

    args = parser.parse_args()

    if args.print_tag:
        print_aruco_tag(args)
        return

    if not args.cam0_json:
        parser.error("--cam0-json is required")
    if not args.cam1_json:
        parser.error("--cam1-json is required")
    if not args.tag_size_m:
        parser.error("--tag-size-m is required")


    cam0_json = Path(args.cam0_json)
    cam1_json = Path(args.cam1_json)

    K0, dist0, w0, h0, intr0 = load_intrinsics(cam0_json)
    K1, dist1, w1, h1, intr1 = load_intrinsics(cam1_json)

    target_id: int | None = None if args.target_id < 0 else args.target_id

    print("=== Camera 0 extrinsics ===")
    cam0_extrinsics = capture_extrinsics_for_camera(
        camera_index=args.cam0_index,
        K=K0,
        dist=dist0,
        img_w=w0,
        img_h=h0,
        tag_size_m=args.tag_size_m,
        aruco_dict_id=args.aruco_dict,
        target_id=target_id,
        window_name="Camera 0",
    )

    print("=== Camera 1 extrinsics ===")
    cam1_extrinsics = capture_extrinsics_for_camera(
        camera_index=args.cam1_index,
        K=K1,
        dist=dist1,
        img_w=w1,
        img_h=h1,
        tag_size_m=args.tag_size_m,
        aruco_dict_id=args.aruco_dict,
        target_id=target_id,
        window_name="Camera 1",
    )

    out_data: Dict[str, Any] = {
        "world_frame": f"aruco_tag_{cam0_extrinsics['detected_id']}",
        "tag_size_m": args.tag_size_m,
        "aruco_dictionary": int(args.aruco_dict),
        "target_id": cam0_extrinsics["detected_id"],  # both should see same ID
        "camera0": {
            "intrinsics": intr0,
            "extrinsics": cam0_extrinsics,
        },
        "camera1": {
            "intrinsics": intr1,
            "extrinsics": cam1_extrinsics,
        },
    }

    out_path = Path(args.output_json)
    with out_path.open("w") as f:
        json.dump(out_data, f, indent=2)

    print(f"Saved full stereo calibration to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
