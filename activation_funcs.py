import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def differential_sigmoid(y):
    return np.multiply(y, (1.0 - y))
