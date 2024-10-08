cmake_minimum_required(VERSION 3.8)
project(mnist_ament_cmake)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/AddThreeInts.srv"
  "srv/MnistSample.srv"
)

ament_export_dependencies(rosidl_default_runtime)

install(PROGRAMS
  scripts/mnist_service.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
