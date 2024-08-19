# Adapted from the TensorFlow tutorials at https://www.tensorflow.org/tutorials/

import os

import tensorflow as tf
import tensorflow.keras.layers as layers

from data import get_data
from model import create_model


print("TensorFlow version: ", tf.version.VERSION)

(x_train, y_train), (x_test, y_test) = get_data()

model = create_model()
print(model.summary())

checkpoint_path = os.path.expanduser("~/mnist_tf_ros2_checkpoints/checkpoint.weights.h5")
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1,
)

model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_data=(x_test, y_test),
    callbacks=[cp_callback],
)

print(os.listdir(checkpoint_dir))

