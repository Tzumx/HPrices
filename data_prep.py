import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from matplotlib import pyplot as plt

SAMPLE = "sample_submission.csv"
TEST_SAMPLE = "test.csv"
TRAIN_SAMPLE = "train.csv"

# print(train_sample.head())
# print(train_data.info())
# print(train_data.describe())
# print(train_data.loc[:,'BsmtFinType1'].value_counts())

def data_norm(sample):
    train_sample = sample.copy(deep=True)
    target_sample = pd.DataFrame([])
    if 'SalePrice' in train_sample.columns: target_sample['SalePrice'] = train_sample.pop('SalePrice')

    for colum in train_sample.columns.values.tolist():

        if colum in ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                    'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 
                    'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']: 
            train_sample.loc[:,colum].replace(np.nan, 'No', inplace=True)

        if pd.api.types.is_numeric_dtype(train_sample.loc[:,colum]):
            if train_sample.loc[:,colum].count() < train_sample.count().max():
                train_sample.loc[:,colum].fillna(train_sample.loc[:,colum].mean(),inplace=True)
            train_sample.loc[:,colum] = minmax_scale(train_sample.loc[:,colum], (0,100))
        else:
            if train_sample.loc[:,colum].count() < train_sample.count().max():
                train_sample.loc[:,colum].fillna(train_sample.loc[:,colum].mode(),inplace=True)
            train_sample = pd.get_dummies(train_sample, columns=[colum], prefix=colum)
    
    return train_sample, target_sample

def data_filter(X_sample, Y_sample, cor_target = 0.01, chi_target = 250):

    train_sample = X_sample.copy(deep=True)
    target_sample = Y_sample.copy(deep=True)

    cor = train_sample.corrwith(target_sample)
    train_sample_drop = cor[abs(cor)<cor_target].index.values
    train_sample.drop(train_sample_drop, axis=1, inplace=True)

    select = SelectKBest(chi2, k=chi_target)
    fit = select.fit(train_sample, target_sample)
    train_sample_drop = pd.Series(fit.scores_, index=train_sample.columns).nsmallest(len(train_sample.columns)-chi_target).index.values
    train_sample.drop(train_sample_drop, axis=1, inplace=True)

    return train_sample, target_sample

    # print((train_sample.info(max_cols=333)))
if __name__ == "__main__":
    train_data = pd.read_csv(TRAIN_SAMPLE, index_col='Id')
    train_sample, target_sample = data_norm(train_data)
    data_filter(train_sample, target_sample)
