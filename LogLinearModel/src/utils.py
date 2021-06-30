"""
有用的函数
"""
import numpy as np


def softmax(X):
    return np.exp(X) / np.sum(np.exp(X))
