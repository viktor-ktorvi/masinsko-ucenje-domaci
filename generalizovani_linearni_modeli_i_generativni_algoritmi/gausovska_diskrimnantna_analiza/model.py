import os

import numpy as np

from collections import Counter
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import accuracy_score

from data_loading.data_loading import load_transformed_data
from generalizovani_linearni_modeli_i_generativni_algoritmi.logisticka_regresija.visualization import \
    dataset_area_class_visualization


def predict(x, means, sigmas, apriori_ys):
    """
    Predict the classes of samples x using Gaussian discriminant analysis.
    :param x: np.ndarray; shape num_samples x num_features; feature matrix
    :param means: np.ndarray; shape num_classes x num_features x 1; estimated means
    :param sigmas: np.ndarray; shape num_classes x num_features x num_features; estimated covariance matrices
    :param apriori_ys: np.ndarray; shape num_classes; estimated apriori probabilities
    :return: np.ndarray; shape num_samples; multiclass predictions
    """
    n_classes = means.shape[0]
    aposterioris = np.array(
        [multivariate_normal(means[i].squeeze(), sigmas[i]).pdf(x) * apriori_ys[i] for i in range(n_classes)]).squeeze()
    return np.argmax(aposterioris, axis=0)


if __name__ == '__main__':
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    csv_path = os.path.join(__location__, '../multiclass_data.csv')

    X_train_transformed, X_test_transformed, y_train, y_test = load_transformed_data(csv_path, test_size=0.2, norm=True,
                                                                                     dimensionality=2)

    cnt = Counter(y_train.squeeze())

    n_classes = len(cnt)
    n_features = X_train_transformed.shape[1]

    apriori_y_est = np.zeros((n_classes,))
    mean_est = np.zeros((n_classes, n_features, 1))
    sigma_est = np.zeros((n_classes, n_features, n_features))

    for y_i in cnt.keys():
        id = round(y_i)
        apriori_y_est[id] = cnt[y_i] / len(y_train)

        X_y_i = X_train_transformed[y_train.squeeze() == y_i, :]
        mean_est[id, :, :] = np.mean(X_y_i, axis=0).reshape((n_features, 1))
        sigma_est[id, :, :] = np.cov(X_y_i.T)

    y_pred = predict(X_test_transformed, mean_est, sigma_est, apriori_y_est)
    print('Test accuracy = {:2.2f}'.format(accuracy_score(np.round(y_test.squeeze()), y_pred)))

    x_transform = X_train_transformed
    y = y_train
    resolution = (50, 50)

    x_min = np.min(x_transform[:, 0])
    x_max = np.max(x_transform[:, 0])
    y_min = np.min(x_transform[:, 1])
    y_max = np.max(x_transform[:, 1])

    x_range = x_max - x_min
    y_range = y_max - y_min

    x_lim = [-0.1 * x_range + x_min, 0.1 * x_range + x_max]
    y_lim = [-0.1 * y_range + y_min, 0.1 * y_range + y_max]

    x_linspace = np.linspace(*x_lim, resolution[0])
    y_linspace = np.linspace(*y_lim, resolution[1])
    xx, yy = np.meshgrid(x_linspace, y_linspace)

    plt.figure()
    for i in range(n_classes):
        gssn = multivariate_normal(mean_est[i, :, :].squeeze(), sigma_est[i, :, :]).pdf(np.dstack((xx, yy)))

        plt.contour(x_linspace, y_linspace, gssn)
        plt.scatter(*[x_transform[np.round(y_train.squeeze()) == i, j] for j in range(x_transform.shape[1])],
                    label="Klasa {:d}".format(i))

    plt.title('Procena raspodele')
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.xlabel('$\hat{x}_1$')
    plt.ylabel('$\hat{x}_2$')
    plt.legend()

    dataset_area_class_visualization(x_transform, y,
                                     lambda background_points: predict(background_points, mean_est, sigma_est,
                                                                       apriori_y_est),
                                     resolution=resolution)  # class areas

    plt.show()
