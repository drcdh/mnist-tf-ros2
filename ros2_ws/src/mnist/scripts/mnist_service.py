#!/usr/bin/env python3
import random

import numpy as np

import rclpy
import rclpy.node

from mnist.srv import MnistSample

from mnist_tf.data import get_data
from mnist_tf.model import load_model


class MnistService(rclpy.node.Node):
    def __init__(self):
        super().__init__("mnist_service")
        self.srv = self.create_service(MnistSample, "mnist_sample", self.mnist_sample_callback)
        _, self.data = get_data()
        self.model = load_model()

    def mnist_sample_callback(self, request, response):
        self.get_logger().info("Incoming request")
        random_pick = random.randrange(len(self.data[0]))
        x, y = self.data[0][random_pick], self.data[1][random_pick]
        y_pred = np.argmax(self.model.predict(x.reshape((1, 28*28)))[0])
        response.msg = f"Sample {random_pick} is a {y}, and the model predicted {y_pred}"
        return response

def main(args=None):
    rclpy.init(args=args)
    mnist_service = MnistService()
    rclpy.spin(mnist_service)
    rclpy.shutdown()

if __name__ == "__main__":
    random.seed(1959)
    main()

