#!/usr/bin/env zsh

set -e

RUN_DIR="${PWD}"

ros2 run cv_hand_controller cv_hand_controller_node.py \
  --ros-args \
  -p model_path:="${RUN_DIR}/model/hand_landmarker.task" \
  -p camera_index:=0 \
  -p camera_fps:=30.0 \
  -p detection_period_ms:=100.0 \
  -p output_dir:="/tmp/test"
  
