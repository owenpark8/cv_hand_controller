#!/usr/bin/env python3

import rclpy
from cv_hand_controller.node import main

if __name__ == "__main__":
    rclpy.init()
    main()
    rclpy.shutdown()

