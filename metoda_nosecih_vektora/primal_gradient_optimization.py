import os
import torch

from cvxopt import matrix, solvers
import numpy as np

from scipy.linalg import block_diag
from skimage import filters
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt

from data_loading.data_loading import load_data
from generalizovani_linearni_modeli_i_generativni_algoritmi.report_visual import dataset_visualization
from generalizovani_linearni_modeli_i_generativni_algoritmi.logisticka_regresija.visualization import \
    dataset_area_class_visualization


def get_weights(a, y_true, x):
    return torch.sum(a * y_true * x, dim=0)


def get_bias(a, y_true, x):
    return y_true[-1] - torch.sum(a * y_true * x * x[-1].T)


def project(x, w, b):
    return x @ w + b


def predict(x, w, b):
    """
    Return class predictions.
    :param X: np.ndarray; shape num_samples x num_features; feature matrix.
    :param w: np.ndarray; shape num_features x 1; classifier weights
    :param w: np.ndarray; shape 1 x 1; classifier bias
    :return: np.ndarray; shape num_samples x 1; predictions in {-1, 1}
    """
    return np.sign(project(x, w, b))


def lagrangian(a, ksi, mu, y_true, x, C):
    w = get_weights(a, y_true, x)
    b = y_true[-1] - torch.sum(a * y_true * x * x[-1].T)

    return 0.5 * w.T @ w + C * torch.sum(ksi) - torch.sum(a * (y_true * project(x, w, b) - 1 + ksi)) - torch.sum(mu * ksi)


def add_support_vector_visualization(current_axis, x_transform, y, support_vector_ids, slack_vars):
    support_vectors = x_transform[support_vector_ids]
    support_vector_labels = y[support_vector_ids].squeeze()
    support_vector_slacks = slack_vars[support_vector_ids].squeeze()

    for clss in np.unique(support_vector_labels):
        current_axis.scatter(*[support_vectors[support_vector_labels == clss, i] for i in range(2)],
                             facecolor='none',
                             edgecolor='lime',
                             linewidths=2,
                             label='noseći vektori')

    for i in range(len(support_vector_slacks)):
        current_axis.annotate(r"$\xi = ${:.3f}".format(support_vector_slacks[i]), (support_vectors[i, 0], support_vectors[i, 1]))

    # suppress duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())


def train(x, y, C, epochs=100):
    num_samples = x.shape[0]
    a = torch.nn.Parameter(torch.ones(num_samples, 1), requires_grad=True)
    ksi = torch.nn.Parameter(torch.ones(num_samples, 1), requires_grad=True)
    mu = torch.nn.Parameter(torch.ones(num_samples, 1), requires_grad=True)

    x = torch.tensor(x, requires_grad=True)
    y = torch.tensor(y, requires_grad=True)

    optimizer = torch.optim.Adam([a, ksi, mu], lr=0.00001)

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = lagrangian(a.data, ksi.data, mu.data, y, x, C)
        loss.backward()
        optimizer.step()

    return get_weights(a.data, y, x), get_bias(a.data, y, x), a.data, ksi.data


if __name__ == '__main__':
    test_size = 0.2
    random_state = 357862
    C = 1  # soft margin hyperparameter

    for i in range(10):
        x, y = make_blobs(n_samples=1000, centers=2, n_features=2)
        y = np.sign(y - 0.5)[:, np.newaxis]

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

        pipe = Pipeline([('scaler', StandardScaler())])  # normalization
        pipe.fit(X_train)

        weights, bias, svm_inequality_lm, slack_vars = train(pipe.transform(X_train), y_train, C)
        weights = weights.detach().cpu().numpy()
        bias = bias.detach().cpu().numpy()
        svm_inequality_lm = svm_inequality_lm.detach().cpu().numpy()
        slack_vars = slack_vars.detach().cpu().numpy()

        # threshold = filters.threshold_otsu(svm_inequality_lm)
        threshold = 0.01
        support_vector_ids = (svm_inequality_lm > threshold).squeeze()

        y_pred = predict(pipe.transform(X_test), weights, bias)
        print("Test accuracy = {:2.2f}".format(accuracy_score(y_test, y_pred)))

        dataset_area_class_visualization(pipe.transform(X_train), y_train,
                                         predict_foo=lambda background_points: predict(background_points, weights, bias),
                                         resolution=(100, 100))

        add_support_vector_visualization(plt.gca(), pipe.transform(X_train), y_train, support_vector_ids, slack_vars)

        current_axis = plt.gca()

    plt.show()

    kjkszpj = None
