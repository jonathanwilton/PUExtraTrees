import numpy as np
from sklearn.datasets import fetch_openml
from trees import PUExtraTrees
import matplotlib.pyplot as plt

X, y = fetch_openml('mnist_784', return_X_y = True, as_frame = False)
y = y.astype(np.int8)
y[y != 0] = -1 # 1 -> 9 forms N class
y[y == 0] = 1 # 0 forms P class
pi = (y == 1).mean()
X_train, y_train, X_test, y_test = X[:60000], y[:60000], X[60000:], y[60000:]

n_p = 1000
positive_indices = np.random.choice(np.where(y_train == 1)[0], size = n_p, replace = False)
P = X_train[positive_indices]
U = X_train.copy()

g = PUExtraTrees()
g.fit(P=P, U=U, pi = pi)
predictions = g.predict(X_test)
print('Accuracy', (predictions == y_test).mean())
