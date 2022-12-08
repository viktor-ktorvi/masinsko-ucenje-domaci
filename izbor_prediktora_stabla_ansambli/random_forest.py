import os
import wandb

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from data_loading.data_loading import load_data

if __name__ == '__main__':
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    csv_path_1 = os.path.join(__location__, 'data_2.csv')  # data path

    wandb.init(project="mu-domaci-logisticka-regresija", config=os.path.join(__location__, 'random-forest-config-defaults.yaml'))

    x, y = load_data(csv_path_1)  # load data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=wandb.config.random_state)

    random_forest_clf = RandomForestClassifier(n_estimators=wandb.config.n_estimators, max_depth=wandb.config.max_depth, max_features=wandb.config.max_features, random_state=wandb.config.random_state)
    random_forest_clf.fit(x_train, y_train.squeeze())

    y_pred = random_forest_clf.predict(x_test)

    wandb.log({'accuracy': accuracy_score(y_test, y_pred)})
