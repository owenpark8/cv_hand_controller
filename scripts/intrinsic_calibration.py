#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]
Mat = NDArray[np.float64]

def create_charuco_board(
    args: argparse.Namespace,
) -> Tuple[Any, Any]:
    aruco_dict = cv2.aruco.getPredefinedDictionary(args.aruco_dict)

    board = cv2.aruco.CharucoBoard(
        size=(args.squares_x, args.squares_y),
        squareLength=args.square_length,
        markerLength=args.marker_length,
        dictionary=aruco_dict,
    )

    return aruco_dict, board


def print_board(args: argparse.Namespace) -> None:
    _, board = create_charuco_board(args)

    square_px: int = args.square_px
    margin_px: int = args.margin_px

    board_w_px: int = args.squares_x * square_px
    board_h_px: int = args.squares_y * square_px

    img_w_px: int = board_w_px + 2 * margin_px
    img_h_px: int = board_h_px + 2 * margin_px

    board_img: Image = board.generateImage(
        outSize=(img_w_px, img_h_px),
        marginSize=margin_px,
        borderBits=1,
    )

    out_path = Path(args.output_board)
    ok: bool = cv2.imwrite(str(out_path), board_img)
    if not ok:
        raise RuntimeError(f"Failed to write board image to {out_path}")

    print(
        f"ChArUco board written to {out_path.resolve()}\n"
        f"  image size: {img_w_px} x {img_h_px} px\n"
        f"  squares:    {args.squares_x} x {args.squares_y}\n"
        f"  square_px:  {square_px}\n"
        f"  margin_px:  {margin_px}"
    )


def calibrate(args: argparse.Namespace) -> None:
    aruco_dict, board = create_charuco_board(args)

    charuco_params = cv2.aruco.CharucoParameters()
    detector_params = cv2.aruco.DetectorParameters()
    charuco_detector = cv2.aruco.CharucoDetector(
        board, charuco_params, detector_params
    )

    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {args.camera_index}")

    if args.width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    all_charuco_corners: List[Mat] = []
    all_charuco_ids: List[Mat] = []
    image_size: Optional[Tuple[int, int]] = None

    print("Controls:")
    print("  c = capture frame")
    print("  q = finish & calibrate")

    last_charuco_corners: Optional[Mat] = None
    last_charuco_ids: Optional[Mat] = None
    last_marker_corners: Optional[List[Mat]] = None
    last_marker_ids: Optional[Mat] = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray: Image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            h, w = gray.shape
            image_size = (w, h)

        charuco_corners, charuco_ids, marker_corners, marker_ids = (
            charuco_detector.detectBoard(gray)
        )

        last_charuco_corners = charuco_corners
        last_charuco_ids = charuco_ids
        last_marker_corners = marker_corners
        last_marker_ids = marker_ids

        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)

        n: int = 0 if charuco_ids is None else len(charuco_ids)
        if n > 4 and charuco_corners is not None:
            cv2.aruco.drawDetectedCornersCharuco(
                frame, charuco_corners, charuco_ids
            )
            cv2.putText(
                frame,
                f"ChArUco corners: {n}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        cv2.putText(
            frame,
            f"Samples: {len(all_charuco_corners)}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("ChArUCo Calibration", frame)
        key: int = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            if (
                last_charuco_corners is None
                or last_charuco_ids is None
                or len(last_charuco_ids) <= 4
            ):
                print("Skipping capture: insufficient ChArUco corners")
                continue

            all_charuco_corners.append(last_charuco_corners)
            all_charuco_ids.append(last_charuco_ids)
            print(
                f"Captured view {len(all_charuco_corners)} "
                f"({len(last_charuco_ids)} corners)"
            )

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if image_size is None:
        raise RuntimeError("No frames captured")

    if len(all_charuco_corners) < args.min_samples:
        raise RuntimeError(
            f"Only {len(all_charuco_corners)} samples collected "
            f"(need {args.min_samples})"
        )

    print("Running calibration...")

    rms, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_charuco_corners,
        charucoIds=all_charuco_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
    )

    result: Dict[str, Any] = {
        "image_width": image_size[0],
        "image_height": image_size[1],
        "camera_matrix": K.tolist(),
        "distortion_coefficients": dist.ravel().tolist(),
        "rms_reprojection_error": float(rms),
        "board": {
            "squares_x": args.squares_x,
            "squares_y": args.squares_y,
            "square_length_m": args.square_length,
            "marker_length_m": args.marker_length,
            "aruco_dictionary": int(args.aruco_dict),
        },
        "per_view_errors": [],
    }

    out_path = Path(args.output_json)
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved intrinsics to: {out_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ChArUco board generator + camera intrinsics calibration (OpenCV 4.12+)"
    )

    # board params
    parser.add_argument("--squares-x", type=int, default=5)
    parser.add_argument("--squares-y", type=int, default=7)
    parser.add_argument("--square-length", type=float, default=0.04)
    parser.add_argument("--marker-length", type=float, default=0.02)
    parser.add_argument(
        "--aruco-dict",
        type=int,
        default=cv2.aruco.DICT_4X4_50,
    )

    # board printing
    parser.add_argument("--print-board", action="store_true")
    parser.add_argument(
        "--square-px",
        type=int,
        default=240,
        help="Square size in pixels when printing board",
    )
    parser.add_argument(
        "--margin-px",
        type=int,
        default=100,
        help="Margin in pixels around the board when printing board",
    )
    parser.add_argument(
        "--output-board",
        type=str,
        default="charuco_board.png",
        help="Output PNG path for the generated board",
    )

    # calibration
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument(
        "--output-json",
        type=str,
        default="camera_intrinsics.json",
        help="Output JSON path for calibrated intrinsics",
    )

    args = parser.parse_args()

    if args.print_board:
        print_board(args)
    else:
        calibrate(args)


if __name__ == "__main__":
    main()
