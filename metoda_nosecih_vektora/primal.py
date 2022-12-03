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


class SVMPrimal:
    def __init__(self, x_train, y_train, C):
        self.b = None
        self.w = None
        self.sv_bool = None

        self.C = C
        self.sv_alpha = None
        self.bias = None

        self.train(x_train, y_train)

    def train(self, x_train, y_train):
        num_samples, num_features = x_train.shape
        P = block_diag(np.eye(num_features), np.zeros((num_samples + 1, num_samples + 1)))
        q = np.vstack((np.zeros((num_features + 1, 1)), self.C * np.ones((num_samples, 1))))

        G1 = np.hstack((-y_train * x_train, -y_train, -np.eye(num_samples)))
        h1 = -np.ones((num_samples, 1))

        G2 = np.hstack((np.zeros((num_samples, num_features + 1)), -np.eye(num_samples)))
        h2 = np.zeros((num_samples, 1))

        G = np.vstack((G1, G2))
        h = np.vstack((h1, h2))

        solvers.options['show_progress'] = False
        solution = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))

        self.w = np.array(solution['x'][:num_features]).squeeze()
        self.b = np.array(solution['x'][num_features])

        self.sv_alpha = np.array(solution['z'][:num_samples])
        self.sv_bool = self.sv_alpha > 1e-5

    def predict(self, x):
        return np.sign(x @ self.w + self.b)
