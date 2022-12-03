import os

import numpy as np

from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt

from data_loading.data_loading import load_data
from generalizovani_linearni_modeli_i_generativni_algoritmi.report_visual import dataset_visualization
from generalizovani_linearni_modeli_i_generativni_algoritmi.logisticka_regresija.visualization import \
    dataset_area_class_visualization


def linear_inference(x, w, b):
    """
    Linear inference.
    :param x: np.ndarray; shape num_samples x num_features; feature matrix.
    :param w: np.ndarray; shape num_features x 1; classifier weights
    :param b: np.ndarray; shape 1 x 1; classifier bias
    :return: np.ndarray; shape num_samples x 1; float classifier output
    """
    return x @ w + b


def get_slack_vars(y_true, y_out):
    """
    Calculate slack variables. Zero if correctly classified; distance to correct margin otherwise.
    :param y_true: np.ndarray; shape num_samples x 1; ground truth in {-1, 1}
    :param y_out: np.ndarray; shape num_samples x 1; classifier output
    :return: np.ndarray; shape num_samples x 1; slack variables
    """
    slack_vars = np.zeros(y_true.shape)
    id_slack_nonzero = y_true * y_out < 0
    slack_vars[id_slack_nonzero] = np.abs(y_true[id_slack_nonzero] - y_out[id_slack_nonzero])

    return slack_vars


def lagrangian(y_out, y_true, w, C, slacks, a, mu):
    """
    Primal lagrangian.
    :param y_out: np.ndarray; shape num_samples x 1; classifier output
    :param y_true: np.ndarray; shape num_samples x 1; ground truth in {-1, 1}
    :param w: np.ndarray; shape num_features x 1; classifier weights
    :param C: float; soft margin hyperparameter
    :param slacks: np.ndarray; shape num_samples x 1; slack vairables
    :param a: np.ndarray; shape num_samples x 1; svm inequality constraints lagrange multipliers
    :param mu: np.ndarray; shape num_samples x 1; soft margin inequality constraints lagrange multipliers
    :return: float; primal lagrangian
    """
    return 0.5 * np.sum(w ** 2) + C * np.sum(slacks) - np.sum(a * (y_true * y_out - 1 + slacks)) - np.sum(
        mu * slacks)


def extract_solution(solution, id_ranges):
    """
    Extract elements from the lagrangian optimization solution vector according to id_ranges
    :param solution: np.ndarray; shape num_variables x 1; optimization vector
    :param id_ranges: dict; stores indices of elements of the optimization vector
    :return: tuple(np.ndarray); elements
    """
    slacks = solution[id_ranges['slack_vars'][0]:id_ranges['slack_vars'][1]][:, np.newaxis]
    a = solution[id_ranges['a'][0]:id_ranges['a'][1]][:, np.newaxis]
    mu = solution[id_ranges['mu'][0]:id_ranges['mu'][1]][:, np.newaxis]

    return a, mu, slacks


def calc_weights_and_bias(a, y_true, x_train_data):
    """
    Apply formula for calculating SVM weights and bias from the lagrange multipliers.
    :param a: np.ndarray; shape num_samples x 1; svm inequality constraints lagrange multipliers
    :param y_true: np.ndarray; shape num_samples x 1; ground truth in {-1, 1}
    :param x_train_data: np.ndarray; shape num_samples x num_features; feature matrix.
    :return: tuple(np.ndarray); weights and bias
    """
    w = np.sum(a * y_true * x_train_data, axis=0)[:, np.newaxis]

    at = a * y_true
    xmTxn = np.einsum('ij, jk->ik', x_train_data, x_train_data.T)
    b = np.mean(y_true - np.einsum('i, ii->i', at.squeeze(), xmTxn)[:, np.newaxis]).reshape(1, 1)

    return w, b


def get_weights_and_bias(solution_vector, id_ranges, x_train_data, y_true):
    """
    Get weights and bias from the solution vector
    :param solution_vector: np.ndarray; shape num_variables x 1; optimization vector
    :param id_ranges: dict; stores indices of elements of the optimization vector
    :param x_train_data: np.ndarray; shape num_samples x num_features; feature matrix.
    :param y_true: np.ndarray; shape num_samples x 1; ground truth in {-1, 1}
    :return: tuple(np.ndarray); weights and bias
    """
    a, _, _ = extract_solution(solution_vector, id_ranges)

    return calc_weights_and_bias(a, y_true, x_train_data)


def min_fun(optimization_vector, C, x_train_data, y_true, id_ranges):
    """
    Function to calculate the lagrangian from the data and optimization vector.
    :param optimization_vector: np.ndarray; shape num_variables x 1; optimization vector
    :param C: float; soft margin hyperparameter
    :param x_train_data: np.ndarray; shape num_samples x num_features; feature matrix.
    :param y_true: np.ndarray; shape num_samples x 1; ground truth in {-1, 1}
    :param id_ranges: dict; stores indices of elements of the optimization vector
    :return: float; lagrangian
    """
    a, mu, slacks = extract_solution(optimization_vector, id_ranges)

    w, b = calc_weights_and_bias(a, y_true, x_train_data)

    y_out = linear_inference(x_train_data, w, b)

    return lagrangian(y_out, y_true, w, C, slacks, a, mu)


def get_initial_solution(x_train, y_train):
    """
    Return the initial solution to the optimization problem.
    :param x_train: np.ndarray; shape num_samples x num_features; feature matrix.
    :param y_train: np.ndarray; shape num_samples x 1; ground truth in {-1, 1}
    :return: tuple(np.ndarray, dict)
    """
    a = np.ones((x_train.shape[0], 1))  # margin lagrange multipliers
    mu = np.ones((x_train.shape[0], 1))  # slack lagrange multipliers

    init_weights, init_bias = calc_weights_and_bias(a, y_train, x_train)

    classifier_out = linear_inference(x_train, init_weights, init_bias)

    slacks = get_slack_vars(y_train, classifier_out)  # soft margin slack variables

    variables = {'slack_vars': slacks, 'a': a, 'mu': mu}
    index_ranges = {'slack_vars': None, 'a': None, 'mu': None}  # how to index the solution vector

    init_solution = np.empty((0, 1))

    # create init solution vector and remember how to index it
    for var_key in variables.keys():
        index_ranges[var_key] = [init_solution.shape[0]]
        init_solution = np.concatenate((init_solution, variables[var_key]))
        index_ranges[var_key].append(init_solution.shape[0])

    return init_solution, index_ranges


def predict(X, w, b):
    """
    Return class predictions.
    :param X: np.ndarray; shape num_samples x num_features; feature matrix.
    :param w: np.ndarray; shape num_features x 1; classifier weights
    :param w: np.ndarray; shape 1 x 1; classifier bias
    :return: np.ndarray; shape num_samples x 1; predictions in {-1, 1}
    """
    return np.sign(linear_inference(X, w, b))


def train(x_train, y_train, C, verbose=True, return_opt_variables=False):
    """
    Train an SVM classifier. Return weights and bias.
    :param x_train: np.ndarray; shape num_samples x num_features; feature matrix.
    :param y_train: np.ndarray; shape num_samples x 1; ground truth in {-1, 1}
    :param C: float; soft margin hyperparameter
    :param verbose: boolean; to print outcome or not
    :return: tuple(np.ndarray); weights and bias
    """
    initial_solution, index_ranges = get_initial_solution(x_train, y_train)

    solution_obj = minimize(
        fun=lambda x_var: min_fun(x_var, C=C, x_train_data=x_train, y_true=y_train,
                                  id_ranges=index_ranges),
        x0=initial_solution.squeeze())
    if verbose:
        print('success' if solution_obj.success else 'no success')
        print(solution_obj.message)

    if return_opt_variables:
        w, b = get_weights_and_bias(solution_obj.x, index_ranges, x_train, y_train)
        a, mu, slacks = extract_solution(solution_obj.x, index_ranges)

        return w, b, a, slacks

    return get_weights_and_bias(solution_obj.x, index_ranges, x_train, y_train)


if __name__ == '__main__':
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    csv_path = os.path.join(__location__, 'svmData.csv')

    test_size = 0.2
    random_state = 357862
    C_hp = 0.1  # soft margin hyperparameter

    x, y = load_data(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    pipe = Pipeline([('scaler', StandardScaler())])  # normalization
    pipe.fit(X_train)

    X_train_transformed = pipe.transform(X_train)

    # dataset_visualization(pipe.transform(X_train_transformed), y_train)

    # weights, bias, a, slacks = train(X_train_transformed, y_train, C=C_hp, return_opt_variables=True)

    X_transformed = pipe.transform(x)

    dataset_area_class_visualization(X_transformed, y,
                                     predict_foo=lambda background_points: predict(background_points, weights, bias),
                                     resolution=(50, 50))
    plt.show()

    kjkszpj = None
