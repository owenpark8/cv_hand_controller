#!/usr/bin/env python3

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from geometry_msgs.msg import Vector3
from cv_hand_controller.msg import FingerPoints
from mrover.msg import IK

class IKBridgeNode(Node):
    def __init__(self):
        super().__init__("ik_bridge")

        self.declare_parameter("input_topic", "/triangulated_finger_points")
        self.declare_parameter("output_topic", "/hand_ik")

        # coordinate Mapping (source -> target)
        # determines which input axis maps to the output x, y, z BEFORE scaling/offset
        # options: "x", "y", "z"
        # example: if src_map_x is "y", then output.x = input.y
        descriptor = ParameterDescriptor(dynamic_typing=True)
        # the above descriptor allows string or bool (YAML parses 'y' as True)
        self.declare_parameter("src_map_x", "x", descriptor)
        self.declare_parameter("src_map_y", "y", descriptor)
        self.declare_parameter("src_map_z", "z", descriptor)

        self.declare_parameter("arm_offset_x", 0.0)
        self.declare_parameter("arm_offset_y", 0.0)
        self.declare_parameter("arm_offset_z", 0.0)
        self.declare_parameter("arm_scale_x", 1.0)
        self.declare_parameter("arm_scale_y", 1.0)
        self.declare_parameter("arm_scale_z", 1.0)

        in_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        out_topic = self.get_parameter("output_topic").get_parameter_value().string_value

        axis_lookup = {'x': 0, 'y': 1, 'z': 2}
        try:
            map_x_str = self._get_axis_param("src_map_x")
            map_y_str = self._get_axis_param("src_map_y")
            map_z_str = self._get_axis_param("src_map_z")
            
            self.map_indices = np.array([
                axis_lookup[map_x_str],
                axis_lookup[map_y_str],
                axis_lookup[map_z_str]
            ], dtype=int)
        except KeyError as e:
            self.get_logger().error(f"Invalid axis mapping parameter: {e}. Must be 'x', 'y', or 'z'. Defaulting to identity.")
            self.map_indices = np.array([0, 1, 2], dtype=int)

        self.offset = np.array([
            self.get_parameter("arm_offset_x").value,
            self.get_parameter("arm_offset_y").value,
            self.get_parameter("arm_offset_z").value
        ], dtype=np.float64)

        self.scale = np.array([
            self.get_parameter("arm_scale_x").value,
            self.get_parameter("arm_scale_y").value,
            self.get_parameter("arm_scale_z").value
        ], dtype=np.float64)

        self.m_sub = self.create_subscription(
            FingerPoints,
            in_topic,
            self._callback,
            10
        )
        
        self.m_pub = self.create_publisher(
            IK,
            out_topic,
            10
        )

        self.get_logger().info(f"IK Bridge initialized. Listening on {in_topic}, publishing to {out_topic}")
        self.get_logger().info(f"Transform config: Offset={self.offset}, Scale={self.scale}")

    def _callback(self, msg: FingerPoints):
        p_thumb = np.array(msg.thumb_points, dtype=np.float64)
        p_index = np.array(msg.index_points, dtype=np.float64)

        if np.any(np.isnan(p_thumb)) or np.any(np.isnan(p_index)):
            self.get_logger().debug("Received NaN in finger points. Skipping.")
            return

        center_raw = (p_thumb + p_index) / 2.0

        # use advanced indexing to reorder [x, y, z] -> [src_x, src_y, src_z]
        # rxample: if map is [2, 0, 1] (z, x, y), then new[0]=old[2], new[1]=old[0]...
        center_mapped = center_raw[self.map_indices]

        center_arm = (center_mapped * self.scale) + self.offset

        out_msg = IK()
        out_msg.pos = Vector3(x=center_arm[0], y=center_arm[1], z=center_arm[2])
        out_msg.pitch = 0.0
        out_msg.roll = 0.0

        self.m_pub.publish(out_msg)

    def _get_axis_param(self, param_name: str) -> str:
        val = self.get_parameter(param_name).value
        
        # if YAML parsed 'y' as True, return 'y'
        if isinstance(val, bool) and val is True:
            return 'y'
            
        return str(val).lower().strip()

def main(args=None):
    rclpy.init(args=args)
    node = IKBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
