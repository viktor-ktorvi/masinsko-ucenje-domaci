import os

import cvxpy as cp
import numpy as np

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


def predict(X, w, b):
    """
    Return class predictions.
    :param X: np.ndarray; shape num_samples x num_features; feature matrix.
    :param w: np.ndarray; shape num_features x 1; classifier weights
    :param w: np.ndarray; shape 1 x 1; classifier bias
    :return: np.ndarray; shape num_samples x 1; predictions in {-1, 1}
    """
    return np.sign(X @ w + b)


def train_svm(x_train, y_train, C, verbose=False):
    m = x_train.shape[0]
    n = x_train.shape[1]

    w = cp.Variable((n, 1))
    b = cp.Variable()
    ksi = cp.Variable((m, 1))

    objective = cp.Minimize(0.5 * cp.square(cp.norm(w)) + C * cp.sum(ksi))
    constraints = [cp.matmul(y_train, (x_train @ w + b)) + ksi >= 1, ksi >= 0]
    prob = cp.Problem(objective, constraints)
    prob.solve()

    if verbose:
        if prob.status == 'optimal':
            print("Optimization successful")
        else:
            print("Optimization unsuccessful")

    return w.value, b.value, constraints[0].dual_value, ksi.value


if __name__ == '__main__':
    # __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    # csv_path = os.path.join(__location__, 'svmData.csv')

    # x, y = load_data(csv_path)

    test_size = 0.2
    random_state = 357862
    C = 10  # soft margin hyperparameter

    for i in range(10):
        x, y = make_blobs(n_samples=100, centers=2, n_features=2)
        y = np.sign(y - 0.5)

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

        pipe = Pipeline([('scaler', StandardScaler())])  # normalization
        pipe.fit(X_train)

        weights, bias, a_lagrange_mul, slack_vars = train_svm(pipe.transform(X_train), y_train, C, verbose=True)

        y_pred = predict(pipe.transform(X_test), weights, bias)
        print("Test accuracy = {:2.2f}".format(accuracy_score(y_test, y_pred)))

        dataset_area_class_visualization(pipe.transform(x), y,
                                         predict_foo=lambda background_points: predict(background_points, weights,
                                                                                       bias),
                                         resolution=(50, 50))

    plt.show()

    kjkszpj = None
