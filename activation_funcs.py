import numpy as np
from typing import Union


def sigmoid(x):
    """
    Sigmoid関数の出力値を計算

    Parameters
    ----------
    x : float or numpy.array
        入力値
    """
    return 1.0 / (1.0 + np.exp(-x))


def differential_sigmoid(y):
    """
    Sigmoid関数の微分値を出力値から計算

    Parameters
    ----------
    y : float or numpy.array
        出力値
    """
    return np.multiply(y, (1.0 - y))
