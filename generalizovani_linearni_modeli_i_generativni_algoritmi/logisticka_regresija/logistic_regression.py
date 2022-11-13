import numpy as np

from utils.utils import sigmoid
from data_loading.data_loading import add_bias


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


def predict_multiclass(x, classifiers):
    """
    Perform 'one-vs-rest' logistic regression and round the result.
    :param x: np.ndarray; shape num_samples x num_features; feature matrix
    :param classifiers: List[np.ndarray; shape num_features x 1; classifier coefficients]
    :return: List[int]; shape num_samples; multiclass label predictions
    """
    x_biased = add_bias(x)
    lr_out = np.array([logistic_regression(x_biased, classifiers[i]) for i in range(len(classifiers))]).squeeze()
    return np.argmax(lr_out, axis=0)


def lr_likelihood(h, y):
    """
    Bernoulli's distribution likelihood.
    :param h: np.ndarray; shape num_samples x 1; discriminatory function output
    :param y: np.ndarray; shape num_samples x 1; binary labels
    :return:
    """
    return h ** y * (1 - h) ** (1 - y)
