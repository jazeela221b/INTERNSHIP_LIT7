import numpy as np


def get_linear(n_samples):
    X = np.random.rand(n_samples, 2)
    y = np.where((2 * X[:, 0]) < (X[:, 1]), 1, 0)
    return X, y


def get_nonlinear(w1, w2, n_samples):
    noise = np.random.randn(n_samples) * 3
    offset1 = np.array([np.random.choice([-10, 10]) for _ in range(n_samples)])
    classes1 = np.where(offset1 < 0, 0, 1)
    x1 = np.linspace(0, 10, n_samples)
    y1 = w1 * x1 + noise + offset1
    offset2 = np.array([np.random.choice([-10, 10]) for _ in range(n_samples)])
    classes2 = np.where(offset2 < 0, 0, 1)
    x2 = np.linspace(10, 20, n_samples)
    y2 = w2 * x2 - (w2 - w1) * 10 + noise + offset2
    X = np.hstack((x1, x2))
    y = np.hstack((y1, y2))
    return np.vstack((X, y)).T, np.hstack((classes1, classes2))
