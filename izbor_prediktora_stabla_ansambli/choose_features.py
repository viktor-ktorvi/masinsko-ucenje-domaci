import os

import numpy as np

from sklearn.metrics import accuracy_score
from tqdm import tqdm

from matplotlib import pyplot as plt

from data_loading.data_loading import load_data
from metoda_nosecih_vektora.svm import train_and_predict, SVMSolverType
from utils.validation import repeated_k_fold


def corrcoef(x1, x2):
    """
    Calculate the correlation coefficient between vectors x1 and x2.
    :param x1: np.ndarray; shape num_samples x 1
    :param x2: np.ndarray; shape num_samples x 1
    :return: float; correlation coefficient
    """
    return np.corrcoef(x1.squeeze(), x2.squeeze())[0, 1]


def forward_selection(features, targets, train_and_predict_foo, metric_foo, num_features=None, **kwargs):
    """
    Select features using the forward wrapper algorithm.
    :param features: np.ndarray; shape num_samples x num_features; feature matrix
    :param targets: np.ndarray; shape num_samples x 1; sample labels
    :param train_and_predict_foo: function handle that trains a model and predicts on a test set
    :param metric_foo: metric function to, depending on its output, select features
    :param num_features: int; number of features to select
    :param kwargs: other optional arguments for the repeated_k_fold function
    :return: tuple[List[int], List[float]]; chosen_feature_ids, best_scores
    """
    if num_features is None:
        num_features = features.shape[1]

    chosen_feature_ids = []
    remaining_feature_ids = np.arange(features.shape[1])
    best_scores = []
    for _ in tqdm(range(num_features)):  # for however many features we want
        scores = {}
        for feature_id in remaining_feature_ids:  # add every remaining feature, train a model and assess the target metric
            chosen_features = features[:, chosen_feature_ids + [feature_id]]
            metric_array = repeated_k_fold(chosen_features, targets, train_and_predict_foo, metric_foo, **kwargs)  # training and validation
            scores[feature_id] = np.mean(metric_array)

        best_feature_id = max(scores, key=scores.get)  # find the best feature
        chosen_feature_ids += [best_feature_id]  # choose it
        best_scores += [scores[best_feature_id]]

        id_for_deletion = np.argwhere(remaining_feature_ids == best_feature_id)
        remaining_feature_ids = np.delete(remaining_feature_ids, id_for_deletion)  # remove from remaining features

    return chosen_feature_ids, best_scores


def get_selection_ticks(selected_features):
    """
    Get the subsets in string form
    :param selected_features: List[int]; list of selected features
    :return: List[str]
    """
    ticks = []  # x tick labels in the form of feature id subsets
    for i in range(len(selected_features)):
        ticks.append(str(selected_features[0:i + 1]).replace("[", "").replace("]", ""))

    return ticks


def plot_selection(selected_features, scores, colors, rotation=30, linestyles=':', alpha=1.0, marker='^', markersize=16):
    ticks = get_selection_ticks(selected_features)

    fig = plt.figure()  # plot the selection results
    x_axis = np.arange(len(selected_features))
    plt.xticks(x_axis, ticks, ha="right", size='xx-small', rotation=rotation)  # {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
    for i in range(len(selected_features)):
        plt.vlines(x_axis[i], min(scores), scores[i], colors=colors[selected_features[i], :], linestyles=linestyles, alpha=alpha)
        plt.plot(x_axis[i], scores[i], c=colors[selected_features[i], :], marker=marker, markersize=markersize)
    plt.xlabel('kombinacija prediktora')
    plt.ylabel('tačnost')
    fig.tight_layout()


if __name__ == '__main__':
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    csv_path_1 = os.path.join(__location__, 'data_1.csv')  # data path

    np.random.seed(6122022)  # for reproducibility

    x, y = load_data(csv_path_1)  # load data
    num_features = x.shape[1]

    colors = np.random.rand(num_features, 3)  # plot options
    marker = "^"
    markersize = 16
    stem_alpha = 0.4
    stem_linestyle = ":"

    # get correlation coefficients and sort them
    abs_corr_coefs = np.abs(np.array([corrcoef(x[:, i], y) for i in range(num_features)]))
    corr_sorted_ids = np.argsort(abs_corr_coefs)

    plt.figure()  # plot correlation coefficients
    x_axis = np.arange(num_features)[::-1]
    plt.xticks(x_axis, corr_sorted_ids)
    for i in range(num_features):
        plt.vlines(x_axis[i], min(abs_corr_coefs), abs_corr_coefs[corr_sorted_ids[i]], colors=colors[corr_sorted_ids[i], :], linestyles=stem_linestyle, alpha=stem_alpha)
        plt.plot(x_axis[i], abs_corr_coefs[corr_sorted_ids[i]], marker=marker, c=colors[corr_sorted_ids[i], :], markersize=markersize)

    plt.xlabel('redni broj prediktora')
    plt.ylabel(r'$|\rho(x_i, y)|$')

    svm_labels = np.sign(y - 0.5)  # SVM options
    svm_solver_type = SVMSolverType.Primal
    C = 1.0
    sigma = 0.1


    def train_and_predict_foo(x_train, y_train, x_test):
        """
        Train an SVM and predict on the test set.
        """
        return train_and_predict(x_train, y_train, x_test, svm_solver_type=svm_solver_type, C=C, sigma=sigma)


    accuracies_corr = []
    best_corr_ids = corr_sorted_ids[::-1]
    for i in tqdm(range(len(corr_sorted_ids))):
        features = x[:, best_corr_ids[0:i + 1]]
        accuracies = repeated_k_fold(features, svm_labels, train_and_predict_foo, metric_foo=accuracy_score, n_splits=5, n_repeats=1)
        accuracies_corr.append(np.mean(accuracies))

    plot_selection(list(best_corr_ids), accuracies_corr, colors, rotation=30, linestyles=stem_linestyle, alpha=stem_alpha, marker=marker, markersize=markersize)

    # select features using a wrapper algorithm with a linear SVM classifier
    wrapper_selected_features, accuracies_wrapper = forward_selection(x, svm_labels, train_and_predict_foo=train_and_predict_foo, metric_foo=accuracy_score,
                                                                      num_features=13, n_splits=5, n_repeats=1)

    plot_selection(wrapper_selected_features, accuracies_wrapper, colors, rotation=30, linestyles=stem_linestyle, alpha=stem_alpha, marker=marker, markersize=markersize)
    plt.show()
