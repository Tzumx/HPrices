import data_prep
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from lightgbm import LGBMRegressor

import json

SAMPLE = "sample_submission.csv"
TEST_SAMPLE = "test.csv"
TRAIN_SAMPLE = "train.csv"


def print_metrics(y_preds, y):
    """
    Evaluation performance of prediction method
    """
    print(f'R^2: {r2_score(y_preds, y)}')
    print(f'RMSE: {mean_squared_error(y_preds, y, squared=False)}')
    y_preds[y_preds < 0] = 0
    print(f'RMSLE: {mean_squared_log_error(y_preds, y)}')


def compare_with_target(predict, target):
    """
    Just print compare table and show difference graph
    """
    predict_sample = pd.DataFrame(predict, columns=target.columns.to_list())
    predict_sample['Id'] = target.index
    predict_sample.set_index('Id', inplace=True)
    diff = target.compare(predict_sample)
    diff['diff'] = diff.SalePrice['self'] - diff.SalePrice['other']
    # print(diff)
    print(f"Max mistake = ", diff['diff'].abs().max())

    diff['diff'].plot(kind='density')
    # plt.show()


class RegModelsLearn:
    """
    Class for fit and test regression models
    """

    def __init__(self, train_sample, target_sample) -> None:
        self.models = {}
        try:
            with open('results.json', 'r') as f:
                self.results = json.load(f)
        except:
            self.results = {}
            self.results.update({'results':{}})
        # split data for train & test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            train_sample, target_sample, test_size=0.2)

    def fit(self, model, **kwargs):
        # fit coefficients for model
        model_use = model(**kwargs)
        model_use.fit(self.X_train, self.y_train)
        self.models[type(model_use).__name__] = model_use

    def predict(self, model):
        # predit results for test data
        model_use = self.models[type(model()).__name__]
        predict_array = model_use.predict(self.X_test)
        return predict_array

    def search_best_params(self, model, test_params: dict, **kwargs):
        # search best models parameters
        ridge_grid_search = GridSearchCV(model(**kwargs),
                                         [test_params],
                                         cv=5,
                                         scoring=make_scorer(
                                             mean_squared_error, greater_is_better=False, 
                                             squared=False),
                                         verbose=1)
        ridge_grid_search.fit(self.X_train, self.y_train)

        self.models[type(model()).__name__] = ridge_grid_search.best_estimator_

        # update info in results if got better performance
        need_update = True
        if type(model()).__name__ in self.results['results'].keys():
            if (abs(ridge_grid_search.best_score_) > 
                            self.results['results'][type(model()).__name__]):
                need_update = False
        if need_update:
            self.results.update({type(model()).__name__:ridge_grid_search.best_params_})
            self.results['results'].update({type(model()).__name__:
                                round(abs(ridge_grid_search.best_score_), 5)})
            print(f"Update best results for {type(model()).__name__}")

        return ridge_grid_search.best_params_

    def print_results(self, model, predict_array):
        # print results and show graph
        print(f'{type(model()).__name__}: ')
        print_metrics(predict_array, self.y_test)
        compare_with_target(predict_array, self.y_test)

    def save(self):
        # save best parameters for model
        with open('results.json', 'w', encoding='utf-8') as file:
            json.dump(self.results, file, ensure_ascii=False)


if __name__ == "__main__":
    train_data = pd.read_csv(TRAIN_SAMPLE, index_col='Id')
    test_data = pd.read_csv(TEST_SAMPLE, index_col='Id')

    train_sample, target_sample = data_prep.data_norm(train_data)
    train_sample, target_sample = data_prep.data_filter(
        train_sample, target_sample)

    test_sample, _ = data_prep.data_norm(test_data)
    test_sample = test_sample.drop(
        columns=test_sample.columns.difference(train_sample.columns))
    test_sample[train_sample.columns.difference(
        test_sample.columns).to_list()] = 0

    print("*********************")
    models = RegModelsLearn(train_sample, target_sample)
    
    # Ridge Regression
    best_lr = models.search_best_params(Ridge, {'alpha': [1, 2, 3, 4, 5],
                                                 'solver': ["auto", "sag"]},
                                         random_state=42)
    print(f"Best Ridge params: {best_lr}")
    predict_array_lr = models.predict(Ridge)
    models.print_results(Ridge, predict_array_lr)
    print()

    # k-nearest neighbors
    best_knn = models.search_best_params(KNeighborsRegressor,
                        {'n_neighbors': [
                            1, 2, 3, 4, 6, 8, 10, 15],
                        'weights': ['distance', 'uniform'],
                        'p': [1, 2],
                        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
                        )
    print(f"Best KNN params: {best_knn}")
    predict_array_knn = models.predict(KNeighborsRegressor)
    models.print_results(KNeighborsRegressor, predict_array_knn)    
    print()
    
    # LGBMRegressor
    params = {'learning_rate': [0.025, 0.05, 0.1, 0.3, 0.5],
            'n_estimators': [250, 500, 1000, 1500, 2000],
            'num_leaves': [20, 50, 100, 150, 200],
            'max_depth': [2, 6, 9, 12, 15],
            'min_child_weight': [0.25, 0.5, 1, 1.5],
            'colsample_bytree': [0.1, 0.3, 0.6, 0.8],
            }
    best_lgbm = models.search_best_params(LGBMRegressor,
                                          params,
                                          subsample=0.9)
    print(f"Best LGBMRegressor params: {best_lgbm}")                                          
    predict_array_lgbm = models.predict(LGBMRegressor)
    models.print_results(LGBMRegressor, predict_array_lgbm)
    print()

    models.save()