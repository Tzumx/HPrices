import data_prep
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV

from math import sqrt

SAMPLE = "sample_submission.csv"
TEST_SAMPLE = "test.csv"
TRAIN_SAMPLE = "train.csv"

def print_metrics(y_preds, y):
    """
    Evaluation performance of prediction method
    """
    print(f'R^2: {r2_score(y_preds, y)}')
    print(f'sqrt(MSE): {sqrt(mean_squared_error(y_preds, y))}')
    y_preds[y_preds<0]=0
    print(f'RMSLE: {mean_squared_log_error(y_preds, y)}')

def compare_with_target(predict, target):
    """
    Just print compare table
    """
    predict_sample = pd.DataFrame(predict, columns = target.columns.to_list())
    predict_sample['Id']=target.index
    predict_sample.set_index('Id', inplace=True)
    diff = target.compare(predict_sample)
    diff['diff'] = diff.SalePrice['self'] - diff.SalePrice['other']
    print(diff)
    print(f"Max mistake = ", diff['diff'].abs().max())
    
    diff['diff'].plot(kind='density')
    plt.show()

if __name__ == "__main__":
    train_data = pd.read_csv(TRAIN_SAMPLE, index_col='Id')
    test_data = pd.read_csv(TEST_SAMPLE, index_col='Id')

    train_sample, target_sample = data_prep.data_norm(train_data)
    train_sample, target_sample = data_prep.data_filter(train_sample, target_sample)

    X_train, X_test, y_train, y_test = train_test_split(train_sample, target_sample, test_size=0.2)

    test_sample, _ = data_prep.data_norm(test_data)
    test_sample = test_sample.drop(columns=test_sample.columns.difference(train_sample.columns))
    test_sample[train_sample.columns.difference(test_sample.columns).to_list()] = 0

    lr = Ridge(solver="auto", random_state=42, alpha=5)
    lr.fit(X_train, y_train)
    predict_array_lr = lr.predict(X_test)
    print("Ridge:")
    print_metrics(predict_array_lr, y_test)
    compare_with_target(predict_array_lr, y_test)
    print()
    ridge_grid_search = GridSearchCV(Ridge(random_state=42), 
                                [{'alpha': [1, 2, 3, 4, 5], 'solver': ["auto", "sag"]}],
                                cv=5,
                                scoring=make_scorer(mean_squared_error,greater_is_better=False),
                                verbose=0)
    ridge_grid_search.fit(X_train, y_train)
    print(ridge_grid_search.best_params_)


    knn = KNeighborsRegressor(n_neighbors=8, weights='distance')
    knn.fit(X_train, y_train)
    predict_array_knn = knn.predict(X_test)
    print("KNeighborsRegressor:")
    print_metrics(predict_array_knn, y_test)
    compare_with_target(predict_array_knn, y_test)
    print()

    knn_grid_search = GridSearchCV(KNeighborsRegressor(weights='distance'), 
                                [{'n_neighbors': [1, 2, 3, 4, 6, 8, 10, 15]}],
                                cv=5,
                                scoring=make_scorer(mean_squared_error,greater_is_better=False),
                                verbose=0)
    knn_grid_search.fit(X_train, y_train)
    print(knn_grid_search.best_params_)

