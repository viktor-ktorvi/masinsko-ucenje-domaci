import numpy as np

from utils.utils import sigmoid


def logistic_regression(x, theta):
    """
    Perform logistic regression on data x with weights theta.
    :param x: np.ndarray; shape num_samples x num_features; feature matrix
    :param theta: np.ndarray; shape num_features x 1; classifier coefficients
    :return: np.ndarray; shape num_samples x 1
    """
    return sigmoid(x @ theta)


def predict_binary(x, theta):
    """
    Perform logistic regression with the given weights and round the result.
    :param x: np.ndarray; shape num_samples x num_features; feature matrix
    :param theta: np.ndarray; shape num_features x 1; classifier coefficients
    :return: np.ndarray; shape num_samples x 1
    """
    return np.round(logistic_regression(x, theta))
