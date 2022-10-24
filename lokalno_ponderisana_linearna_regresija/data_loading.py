import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def load_data():
    """
    Load data stored in 'data.csv' where the features are stored in the first n-1 columns and the labels are in the last
    column.
    :return: features and labels (np.ndarray; shape num_samples x num_features, np.ndarray; shape num_samples x 1).
    """
    dataframe = pd.read_csv('data.csv', header=None)
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


if __name__ == '__main__':
    df = pd.read_csv('data.csv', header=None)
    df_numpy = df.to_numpy()

    print('Data stats\n')
    print(df.describe())

    corr = df.corr()

    x, y = load_data()

    n = x.shape[1]

    fig, ax = plt.subplots(n, 1)
    for i in range(n):
        ax[i].scatter(x[:, i], y)

    fig.show()
    plt.show()

    x_biased = add_bias(x)
