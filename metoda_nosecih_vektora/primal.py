import numpy as np

from cvxopt import matrix, solvers
from scipy.linalg import block_diag


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
        self.sv_bool = (self.sv_alpha > 1e-5).squeeze()

    def project(self, x):
        return x @ self.w + self.b

    def predict(self, x):
        return np.sign(self.project(x))
