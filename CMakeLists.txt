cmake_minimum_required(VERSION 3.22)
project(cv_hand_controller)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

ament_python_install_package(cv_hand_controller)

install(PROGRAMS
  scripts/cv_hand_controller_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

