# mnist-tf-ros2
Simple proof-of-concept for using a TensorFlow model in a ROS2 node

# Instructions
Install requirements with `pip install tensorflow h5py` (not an exhaustive list).

Train the model (this will write to `~/mnist_tf_ros2_checkpoints/checkpoint.weights.h5`) with `python /path/to/mnist-tf-ros2/ros2_ws/src/mnist/mnist_tf/train.py`.

Build the ROS2 package with `cd /path/to/mnist-tf-ros2/ros2_ws/ && colcon build`. Source with `source /path/to/mnist-tf-ros2/ros2_ws/install/setup.bash`.

Run the package with `ros2 run mnist mnist_service.py`.

In another terminal, source as before, and then call the service with `ros2 service call /mnist_sample mnist/srv/MnistSample`.
