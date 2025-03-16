import numpy as np
from sklearn.linear_model import Perceptron
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])
model = Perceptron(max_iter=100, eta0=0.1)
model.fit(X, y)
predictions = model.predict(X)
print("Predictions:", predictions)