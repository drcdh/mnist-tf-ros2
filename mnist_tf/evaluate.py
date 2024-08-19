import numpy as np

from data import get_data
from model import load_model

_, (x_test, y_test) = get_data()

model = load_model()
print(model.summary())

_, acc = model.evaluate(x_test, y_test, verbose=2)
print("{:5.2f}%".format(100*acc))

print(x_test.shape)

x, y = x_test[0], y_test[0]
print(x.shape)
y_pred = np.argmax(model.predict(x.reshape((1, 28*28)))[0])
print(f"Predicted {y_pred}, truth is {y}")
