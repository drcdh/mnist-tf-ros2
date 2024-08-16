import os

import tensorflow as tf
import tensorflow.keras.layers as layers


def create_model():
    model = tf.keras.models.Sequential([
        layers.Input((784,)),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(10),
    ])
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


def load_model(checkpoint_path=os.path.expanduser("~/mnist_tf_ros2_checkpoints/checkpoint.weights.h5")):
    model = create_model()
    model.load_weights(checkpoint_path)
    return model

