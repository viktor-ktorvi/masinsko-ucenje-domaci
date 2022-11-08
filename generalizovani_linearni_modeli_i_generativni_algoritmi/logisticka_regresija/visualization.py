import numpy as np

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

from data_loading.data_loading import load_data, add_bias
from utils.utils import normalize
from generalizovani_linearni_modeli_i_generativni_algoritmi.logisticka_regresija.logistic_regression import \
    predict_binary, predict_multiclass


def dataset_visualization(x=None, y=None):
    """
    Reduce dataset dimensionality to 2D and scatter plot.
    :param x: np.ndarray; shape num_samples x num_features; train feature matrix
    :param y: np.ndarray; shape num_samples x 1; multiclass labels
    :return:
    """
    if x is None or y is None:
        x, y = load_data('../multiclass_data.csv')

        x_norm, mean, std = normalize(x)

        pca = PCA(n_components=2)
        pca.fit(x_norm)

        x_ = pca.transform(x_norm)
    else:
        x_ = x

    classes = np.unique(y)

    plt.figure()
    plt.title('Odbirci nakon smanjenja dimenzije i normalizacije')
    for i in range(len(classes)):
        plt.scatter(x_[y.squeeze() == classes[i], 0], x_[y.squeeze() == classes[i], 1], label="Klasa {:d}".format(i))
    plt.xlabel('$\hat{x}_1$')
    plt.ylabel('$\hat{x}_2$')
    plt.legend()


def loss_and_acc_visualization(loggers):
    """
    Plot the loss and accuracy training curves.
    :param loggers: List[Logger]
    :return:
    """
    fig, axs = plt.subplots(1, 2)
    plt.suptitle("Apsolutna greška i tačnost")

    for i in range(len(loggers)):
        axs[0].plot(loggers[i].loss, label="klasa {:d}".format(i))
        axs[1].plot(loggers[i].accuracy, label="klasa {:d}".format(i))

    axs[0].set_title('Apsolutna greška')
    axs[0].set_xlabel('epoha')
    axs[0].set_ylabel('greška')

    axs[1].set_title('Tačnost')
    axs[1].set_xlabel('epoha')
    axs[1].set_ylabel('tačnost')

    axs[0].legend()
    axs[1].legend()


def one_vs_all_visualization(classifiers_2d, x_train_2d, x_test_2d, train_labels, test_labels):
    """
    Plot the trained 'one-vs-rest' classifiers next to each other.
    :param classifiers_2d: List[np.ndarray; shape num_features x 1; classifier coefficients]
    :param x_train_2d: np.ndarray; shape num_samples x num_features; train feature matrix
    :param x_test_2d: np.ndarray; shape num_samples x num_features; train feature matrix
    :param train_labels: np.ndarray; shape num_samples x 1; binary labels
    :param test_labels: np.ndarray; shape num_samples x 1; binary labels
    :return:
    """
    fig, axs = plt.subplots(1, 3, sharey='all')
    plt.suptitle(r"'jedan protiv ostalih' klasifikatori")

    for i in range(len(classifiers_2d)):
        y_pred = predict_binary(add_bias(x_test_2d), classifiers_2d[i]).squeeze()
        acc = accuracy_score(test_labels[i].squeeze(), y_pred)

        axs[i].set_title("Klasa {:d}\nTačnost na test skupu {:2.2f} %".format(i, acc * 100))
        axs[i].scatter(x_train_2d[train_labels[i].squeeze() == 0, 0],
                       x_train_2d[train_labels[i].squeeze() == 0, 1], label='0')
        axs[i].scatter(x_train_2d[train_labels[i].squeeze() == 1, 0],
                       x_train_2d[train_labels[i].squeeze() == 1, 1], label='1')

        x0_domain = np.linspace(start=np.min(x_train_2d[:, 0]), stop=np.max(x_train_2d[:, 0]))
        line = -(classifiers_2d[i][0] + classifiers_2d[i][1] * x0_domain) / classifiers_2d[i][2]
        axs[i].plot(x0_domain, line, label=r'$\theta^T x$', color='r')

        axs[i].set_xlabel('$\hat{x}_1$')
        if i == 0:
            axs[i].set_ylabel('$\hat{x}_2$')
        x_range = np.max(x_train_2d[:, 0]) - np.min(x_train_2d[:, 0])
        y_range = np.max(x_train_2d[:, 1]) - np.min(x_train_2d[:, 1])
        axs[i].set_xlim(-0.1 * x_range + np.min(x_train_2d[:, 0]),
                        0.1 * x_range + np.max(x_train_2d[:, 0]))
        axs[i].set_ylim(-0.1 * y_range + np.min(x_train_2d[:, 1]),
                        0.1 * y_range + np.max(x_train_2d[:, 1]))
        axs[i].legend()


def dataset_area_class_visualization(x_transform, y, classifiers, resolution=(50, 50)):
    """
    Plot data samples and color the background according to what every point would be classified.
    :param x_transform: np.ndarray; shape num_samples x num_features; train feature matrix
    :param y: np.ndarray; shape num_samples x 1; multiclass labels
    :param classifiers: List[np.ndarray; shape num_features x 1; classifier coefficients]
    :param resolution: tuple(int, int); (nrows, ncols)
    :return:
    """
    x_min = np.min(x_transform[:, 0])
    x_max = np.max(x_transform[:, 0])
    y_min = np.min(x_transform[:, 1])
    y_max = np.max(x_transform[:, 1])

    x_range = x_max - x_min
    y_range = y_max - y_min

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution[0]),
                         np.linspace(y_min, y_max, resolution[1]))
    background_points = np.vstack((xx.ravel(), yy.ravel())).T

    fig = plt.figure()
    plt.title('Odbirci i oblasti klasifikacije')
    for clss in np.unique(y):
        plt.scatter(*[x_transform[y.squeeze() == clss, i] for i in range(2)], label="Klasa {:d}".format(int(clss)))

    plt.xlabel('$\hat{x}_1$')
    plt.ylabel('$\hat{x}_2$')
    plt.legend()

    y_pred = predict_multiclass(background_points, classifiers).reshape(xx.shape)
    cmap = ListedColormap(['paleturquoise', 'orange', 'lawngreen'])
    plt.imshow(y_pred,
               extent=[-0.1 * x_range + x_min, 0.1 * x_range + x_max, -0.1 * y_range + y_min, 0.1 * y_range + y_max],
               cmap=cmap,
               alpha=0.3,
               aspect='auto',
               origin='lower')
