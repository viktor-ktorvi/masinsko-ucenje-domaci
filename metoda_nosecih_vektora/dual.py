import argparse
import os

import numpy as np

from cvxopt import matrix, solvers
from enum import IntEnum
from scipy.linalg import block_diag

from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt

from data_loading.data_loading import load_data
from generalizovani_linearni_modeli_i_generativni_algoritmi.logisticka_regresija.visualization import \
    dataset_area_class_visualization
from utils.utils import cross_norm_sqrd
from utils.validation import hyperparameter_search


class SVMDual:
    def __init__(self, x_train, y_train, C, kernel_foo):
        self.sv_bool = None
        self.sv_x = None
        self.sv_y = None
        self.C = C
        self.sv_alpha = None
        self.bias = None
        self.kernel_foo = kernel_foo

        self.train(x_train, y_train)

    def train(self, x_train, y_train):
        num_samples, num_features = x_train.shape
        P = self.kernel_foo(x_train, x_train)  # (y * x) @ (y * x).T
        q = -np.ones((num_samples, 1))
        A = y_train.reshape(1, num_samples)
        b = 0.0

        G = np.vstack((-np.eye(num_samples), np.eye(num_samples)))
        h = np.hstack((np.zeros(num_samples), np.ones(num_samples) * self.C))

        solvers.options['show_progress'] = False
        solution = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b))

        # alfa
        alpha = np.array(solution['x'])

        # support vectors
        self.sv_bool = (alpha > 1e-5).squeeze()
        self.sv_alpha = alpha[self.sv_bool]
        self.sv_x = x_train[self.sv_bool]
        self.sv_y = y_train[self.sv_bool]

        self.bias = np.mean(self.sv_y) - np.sum(
            self.sv_alpha * self.sv_y * self.kernel_foo(self.sv_x, np.mean(self.sv_x, axis=0)[np.newaxis, :]))

    def predict(self, x):
        return np.sign(np.sum(self.sv_alpha * self.sv_y * self.kernel_foo(x, self.sv_x), axis=0) + self.bias)
