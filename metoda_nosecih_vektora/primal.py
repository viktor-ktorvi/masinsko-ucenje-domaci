import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt

from data_loading.data_loading import load_data
from generalizovani_linearni_modeli_i_generativni_algoritmi.report_visual import dataset_visualization

if __name__ == '__main__':
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    csv_path = os.path.join(__location__, 'svmData.csv')

    test_size = 0.2
    random_state = 357862

    x, y = load_data(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    pipe = Pipeline([('scaler', StandardScaler())])
    pipe.fit(X_train)

    dataset_visualization(pipe.transform(X_train), y_train)
    plt.show()
