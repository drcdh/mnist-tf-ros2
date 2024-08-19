# mnist-tf-ros2
Simple proof-of-concept for using a TensorFlow model in ROS2 as part of a service or publisher.

# Instructions
These instructions assume that the current directory is the root of this repository.

Install requirements with `pip install tensorflow h5py` (not an exhaustive list).

Train the model (this will write to `~/mnist_tf_ros2_checkpoints/checkpoint.weights.h5`) with `python ./mnist_tf/train.py`.

Build the ROS2 package with `pushd ./ros2_ws/ && colcon build && popd`. Source with `source ./ros2_ws/install/setup.bash`.

## Publisher
Run the publisher with `ros2 run mnist_pubsub talker`.

## Service
Run the `ament_cmake`-built package with `ros2 run mnist_ament_cmake mnist_service.py` OR run the `ament_python`-built package with `ros2 run mnist_ament_python mnist_service.py`.

In another terminal, source as before, and then call the service with `ros2 service call /mnist_sample mnist/srv/MnistSample`.
