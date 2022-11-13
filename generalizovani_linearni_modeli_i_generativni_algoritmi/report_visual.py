import os

from matplotlib import pyplot as plt

import numpy as np

from data_loading.data_loading import load_transformed_data


def dataset_visualization(x, y):
    """
    Scatter plot od 2D data.
    :param x: np.ndarray; shape num_samples x 2; train feature matrix
    :param y: np.ndarray; shape num_samples x 1; multiclass labels
    :return:
    """

    classes = np.unique(y)

    plt.figure()
    # plt.title('Odbirci nakon smanjenja dimenzije i normalizacije')
    for i in range(len(classes)):
        plt.scatter(x[y.squeeze() == classes[i], 0], x[y.squeeze() == classes[i], 1], label="klasa {:d}".format(i))
    plt.xlabel('$\hat{x}_1$')
    plt.ylabel('$\hat{x}_2$')
    plt.legend()


if __name__ == '__main__':
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    csv_path = os.path.join(__location__, 'multiclass_data.csv')

    X_train_transformed, X_test_transformed, y_train, y_test = load_transformed_data(csv_path,
                                                                                     test_size=0.2, norm=True,
                                                                                     dimensionality=2)

    x_transformed = np.vstack((X_train_transformed, X_test_transformed))
    y = np.vstack((y_train, y_test))

    dataset_visualization(x_transformed, y)
    plt.show()
