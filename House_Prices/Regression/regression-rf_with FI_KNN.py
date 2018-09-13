# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 19:26:14 2018

@author: Ankita
"""
import os
import pandas as pd
#For K Nearest Neighbors
from sklearn import neighbors, feature_selection
from sklearn import preprocessing, ensemble
from sklearn import model_selection, metrics
import matplotlib.pyplot as plt
import seaborn as sns
import math

def get_continuous_columns(df):
    return df.select_dtypes(include=['number']).columns

def get_categorical_columns(df):
    return df.select_dtypes(exclude=['number']).columns

def transform_cat_to_cont(df, features, mappings):
    for feature in features:
        null_idx = df[feature].isnull()
        df.loc[null_idx, feature]= None
        df[feature]= df[feature].map(mappings)
        
def transform_cont_to_cat(df, features):
    for feature in features:
        df[feature] = df[feature].astype('category')
        
def get_missing_features(df):
    total_missing = df.isnull().sum()
    to_delete = total_missing[total_missing > 0]
    return list(to_delete.index)

def filter_features(df, features):
    df.drop(features, axis= 1, inplace = True)
    
def get_heat_map_corr(df):
    corr = df.select_dtypes(include = ['number']).corr()
    sns.heatmap(corr, square= True)
    plt.xticks(rotation= 70)
    plt.yticks(rotation= 70)
    return corr

def get_target_corr(corr, target):
    return corr[target].sort_values(axis=0, ascending = False)

def one_hot_encode(df):
    features = get_categorical_columns(df)
    return pd.get_dummies(df, columns= features)

def rmse(y_orig, y_pred):
    return math.sqrt(metrics.mean_squared_log_error(y_orig, y_pred))

def fit_model(estimator, grid, x_train, y_train):
    grid_estimator= model_selection.GridSearchCV(estimator, grid, scoring= metrics.make_scorer(rmse), cv=10, n_jobs=1)
    grid_estimator.fit(x_train, y_train)
    print(grid_estimator.best_score_)
    print(grid_estimator.score(x_train, y_train))
    return grid_estimator.best_estimator_

def predict(estimator, x_train):
    return estimator.predict(x_train)

#to uncover the mismatch of levels between train and test data
def merge(df1, df2):
    return pd.concat([df1, df2])

def split(df, ind):
    return (df[0:ind], df[ind:])

def viz_cont_cont(df, features, target):
    for feature in features:
        sns.jointplot(x= features, y= target, data= df)
    
def viz_cat_cont(df, features, target):
    for feature in features:
        sns.FacetGrid(df, row=feature,size=8).map(sns.kdeplot, target).add_legend()
        plt.xticks(rotation=45)

def viz_cont(df, features):
    for feature in features:
        sns.distplot(df[feature],kde=False)
        
def feature_selection_from_model(estimator, feature_data, target):
    estimator.fit(feature_data, target)
    
    features = pd.DataFrame({'feature':feature_data.columns, 'importance':estimator.feature_importances_})
    features.sort_values(by=['importance'], ascending=True, inplace=True)
    features.set_index('feature', inplace=True)
    features.plot(kind='barh', figsize=(20, 20))

    fs_model = feature_selection.SelectFromModel(estimator, threshold="mean", prefit=True)
    return features, fs_model.transform(feature_data)

def get_scale_model(df) :
    scaler = preprocessing.StandardScaler()
    scaler.fit(df)
    return scaler

#To get current working directory
os.getcwd()
os.environ['PATH'] = os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin\\'

#To read train data
house_train = pd.read_csv('C:\\Data Science\\data\\House_Prices\\train.csv')
house_train.shape
house_train.info()

#To read test data
house_test = pd.read_csv('C:\\Data Science\\data\\House_Prices\\test.csv')
house_test.shape
house_test.info()
house_test['SalePrice'] = 0

house_data = merge(house_train, house_test)
house_data.shape
house_data.info()

print(get_continuous_columns(house_data))
print(get_categorical_columns(house_data))

#convert numerical columns to categorical type              
features = ['MSSubClass']
transform_cont_to_cat(house_data, features)

#map string categoical values to numbers
ordinal_features = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "GarageQual", "GarageCond", "PoolQC", "FireplaceQu", "KitchenQual", "HeatingQC"]
quality_dict = {None: 0,"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
transform_cat_to_cont(house_data, ordinal_features, quality_dict)

#filter missing data columns
missing_features = get_missing_features(house_data)
filter_features(house_data, missing_features)
house_data.shape
house_data.info()

#explore relation among all continuous features vs saleprice 
corr = get_heat_map_corr(house_train)
get_target_corr(corr, 'SalePrice')

house_train, house_test = split(house_data, house_train.shape[0])
house_train.shape
house_test.shape

features = ['SalePrice']
viz_cont(house_train, features)

#explore relationship of neighborhood to saleprice
target = 'SalePrice'
features = ['Neighborhood']
viz_cat_cont(house_train, features, target)

#explore relationship of livarea and totalbsmt to saleprice
target = ['SalePrice']
features = ['GrLivArea']
#features = ['TotalBsmtSF']
viz_cont_cont(house_train, features, target)

features_to_filter = ['Id']

#do one-hot-encoding for all the categorical features
house_train1 = one_hot_encode(house_train)
house_train1.shape
house_train1.info()

filter_features(house_train1, ['SalePrice'])
X_train = house_train1
y_train = house_train['SalePrice']

X_train.shape
#Step 1 
rf_estimator = ensemble.RandomForestRegressor(n_estimators=50, verbose = 3)
X_train1 = feature_selection_from_model(rf_estimator, X_train, y_train)
#X_train1.shape

scaled_model = get_scale_model(X_train1)
X_train1 = scaled_model.transform(X_train1)

knn_estimator = neighbors.KNeighborsRegressor()
knn_grid = {'n_neighbors':[10, 15]}
model = fit_model(knn_estimator, knn_grid, X_train1, y_train)
