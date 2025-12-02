#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import yaml
from numpy.typing import NDArray

Mat = NDArray[np.float64]


def print_aruco_tag(args: argparse.Namespace) -> None:
    aruco_dict = cv2.aruco.getPredefinedDictionary(args.aruco_dict)

    tag_img = cv2.aruco.generateImageMarker(
        dictionary=aruco_dict,
        id=args.tag_id,
        sidePixels=args.tag_px,
        borderBits=1,
    )

    out_path = Path(args.output_tag)
    if not cv2.imwrite(str(out_path), tag_img):
        raise RuntimeError(f"Failed to write ArUco tag to {out_path}")

    print(
        f"ArUco tag written to {out_path.resolve()}\n"
        f"  tag id:      {args.tag_id}\n"
        f"  dictionary:  {args.aruco_dict}\n"
        f"  resolution:  {args.tag_px} x {args.tag_px} px"
    )


def load_intrinsics(yaml_path: Path) -> Tuple[Mat, Mat, int, int, Dict[str, Any]]:
    """
    Load camera intrinsics from a YAML file of the form:

    image_width: ...
    image_height: ...
    camera_matrix:
      - [fx, 0, cx]
      - [0, fy, cy]
      - [0, 0, 1]
    distortion_coefficients: [...]
    """
    with yaml_path.open("r") as f:
        data: Dict[str, Any] = yaml.safe_load(f)

    width = int(data["image_width"])
    height = int(data["image_height"])

    K = np.asarray(data["camera_matrix"], dtype=np.float64)
    dist = np.asarray(data["distortion_coefficients"], dtype=np.float64).reshape(-1, 1)

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

    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera index {camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_h)

    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(aruco_dict_id),
        cv2.aruco.DetectorParameters(),
    )

    R_wc = t_wc = detected_id = None

    print(f"[cam {camera_index}] Looking for ArUco tag ({tag_size_m} m)")
    print("  c = capture   q = abort")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            if target_id is not None:
                mask = ids.flatten() == target_id
                if not np.any(mask):
                    continue
                corners = [corners[np.where(mask)[0][0]]]
                ids = ids[mask]

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, tag_size_m, K, dist
            )

            rvec, tvec = rvecs[0], tvecs[0]
            detected_id = int(ids[0][0])
            R_wc, _ = cv2.Rodrigues(rvec)
            t_wc = tvec.reshape(3, 1)

            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.drawFrameAxes(frame, K, dist, rvec, tvec, tag_size_m * 0.5)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c") and R_wc is not None:
            break
        if key == ord("q"):
            raise RuntimeError(f"[cam {camera_index}] Capture aborted")

    cap.release()
    cv2.destroyWindow(window_name)

    R_cw = R_wc.T
    t_cw = -R_wc.T @ t_wc

    P = K @ np.hstack((R_wc, t_wc))

    return {
        "camera_index": camera_index,
        "detected_id": detected_id,
        "R_world_to_cam": R_wc.tolist(),
        "t_world_to_cam": t_wc.flatten().tolist(),
        "R_cam_to_world": R_cw.tolist(),
        "t_cam_to_world": t_cw.flatten().tolist(),
        "P_world_to_cam": P.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stereo extrinsics from two cameras relative to an ArUco tag (YAML I/O)"
    )

    parser.add_argument("--cam0-yaml", required=True)
    parser.add_argument("--cam1-yaml", required=True)

    parser.add_argument("--cam0-index", type=int, default=0)
    parser.add_argument("--cam1-index", type=int, default=2)

    parser.add_argument("--tag-size-m", type=float, required=True)
    parser.add_argument("--aruco-dict", type=int, default=cv2.aruco.DICT_4X4_50)
    parser.add_argument("--target-id", type=int, default=-1)

    parser.add_argument(
        "--output-yaml",
        default="stereo_calibration.yaml",
        help="Output YAML file",
    )

    parser.add_argument("--print-tag", action="store_true")
    parser.add_argument("--tag-id", type=int, default=0)
    parser.add_argument("--tag-px", type=int, default=600)
    parser.add_argument("--output-tag", default="aruco_tag.png")

    args = parser.parse_args()

    if args.print_tag:
        print_aruco_tag(args)
        return

    K0, d0, w0, h0, intr0 = load_intrinsics(Path(args.cam0_yaml))
    K1, d1, w1, h1, intr1 = load_intrinsics(Path(args.cam1_yaml))

    target_id = None if args.target_id < 0 else args.target_id

    cam0 = capture_extrinsics_for_camera(
        args.cam0_index, K0, d0, w0, h0,
        args.tag_size_m, args.aruco_dict,
        target_id, "Camera 0"
    )

    cam1 = capture_extrinsics_for_camera(
        args.cam1_index, K1, d1, w1, h1,
        args.tag_size_m, args.aruco_dict,
        target_id, "Camera 1"
    )

    out = {
        "world_frame": f"aruco_tag_{cam0['detected_id']}",
        "tag_size_m": args.tag_size_m,
        "aruco_dictionary": int(args.aruco_dict),
        "target_id": cam0["detected_id"],
        "camera_0": {"intrinsics": intr0, "extrinsics": cam0},
        "camera_1": {"intrinsics": intr1, "extrinsics": cam1},
    }

    out_path = Path(args.output_yaml)
    with out_path.open("w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    print(f"Saved stereo calibration to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
