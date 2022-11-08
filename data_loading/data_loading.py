import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from utils.utils import normalize


def load_data(csv_path):
    """
    Load data stored in csv_path where the features are stored in the first n-1 columns and the labels are in the last
    column.
    :return: features and labels (np.ndarray; shape num_samples x num_features, np.ndarray; shape num_samples x 1).
    """
    dataframe = pd.read_csv(csv_path, header=None)
    dataframe_np = dataframe.to_numpy()

    return dataframe_np[:, :-1], dataframe_np[:, -1].reshape(dataframe_np.shape[0], 1)


def add_bias(x):
    """
    Add ones as the 0th feature.
    :param x: np.ndarray; shape num_samples x num_features
    :return: np.ndarray; shape num_samples x (num_features + 1)
    """
    ones = np.ones((x.shape[0], 1))
    return np.concatenate((ones, x), axis=1)


def load_transformed_data(csv_path, test_size=0.2, norm=False, dimensionality=None, random_state=546315):
    """
    Load data, split it and transform it.
    :param csv_path: string; data path
    :param test_size: float; test set ration
    :param norm: bool; to normalize or not
    :param dimensionality: int or None; dimension to reduce the data to
    :param random_state: int; random seed
    :return: tuple(np.ndarray; shape num_train_samples x num_features, np.ndarray; shape num_test_samples x num_features,
    np.ndarray; shape num_train_samples x 1, np.ndarray; shape num_test_samples x 1)
    """
    x, y = load_data(csv_path)

    # split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    # transform
    if norm:
        X_train_transformed, mean, std = normalize(X_train)
        X_test_transformed = normalize(X_test, mean, std)
    else:
        X_train_transformed = X_train
        X_test_transformed = X_test

    if dimensionality is not None:
        pca = PCA(n_components=dimensionality)  # 2d for visualization purposes
        pca.fit(X_train_transformed)

        X_train_transformed = pca.transform(X_train_transformed)
        X_test_transformed = pca.transform(X_test_transformed)

    return X_train_transformed, X_test_transformed, y_train, y_test


if __name__ == '__main__':
    data_path = '../generalizovani_linearni_modeli_i_generativni_algoritmi/multiclass_data.csv'
    df = pd.read_csv(data_path, header=None)
    df_numpy = df.to_numpy()

    print('Data stats\n')
    print(df.describe())

    corr = df.corr()

    x, y = load_data(data_path)

    n = x.shape[1]

    fig, ax = plt.subplots(n, 1)
    for i in range(n):
        ax[i].scatter(x[:, i], y)

    fig.show()
    plt.show()

    x_biased = add_bias(x)
