cmake_minimum_required(VERSION 3.22)
project(cv_hand_controller)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  msg/FingerPoints.msg
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)

# ament_python_install_package(cv_hand_controller)

install(PROGRAMS
  cv_hand_controller/camera_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY model
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
