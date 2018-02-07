import numpy as np

X = 2 * np.random.rand(100000, 1)
y = 4 + 3 * X + np.random.randn(100000, 1)

X_b = np.c_[np.ones((100000, 1)), X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
