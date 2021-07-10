"""
激活函数,及激活函数导数(用于反向传播)
"""
import numpy as np


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def sigmoid_derivative(X):
    return sigmoid(X)(1 - sigmoid(X))


def relu(X):  # np.maximum()比较每一个元素和0的最大值
    return np.maximum(0, X)


def relu_derivative(X):  # np.int64()会把True转换成1,把False转换成0
    return np.int64(X > 0)


def softmax(X):
    return np.exp(X) / np.sum(np.exp(X))
