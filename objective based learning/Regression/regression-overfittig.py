# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 22:19:35 2018

@author: Ankita
"""
import os
import pandas as pd
from pandas import Series
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import model_selection, metrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

def get_continuous_columns(df):
    return df.select_dtypes(include=['number']).columns

def get_categorical_columns(df):
    return df.select_dtypes(exclude=['number']).columns

def transform_cat_to_cont(df, features, mappings):
    for feature in features:
        null_idx = df[feature].isnull()
        df.loc[null_idx, feature] = None 
        df[feature] = df[feature].map(mappings)

def transform_cont_to_cat(df, features):
    for feature in features:
        df[feature] = df[feature].astype('category')

def get_features_missing_data(df, cutoff):
    total_missing = df.isnull().sum()
    to_delete = total_missing[(total_missing) > cutoff ]
    return list(to_delete.index)

def filter_features(df, features):
    df.drop(features, axis=1, inplace=True)

def get_imputers(df, features):
    all_cont_features = get_continuous_columns(df)
    cont_features = []
    cat_features = []
    for feature in features:
        if feature in all_cont_features:
            cont_features.append(feature)
        else:
            cat_features.append(feature)
    mean_imputer = preprocessing.Imputer()
    mean_imputer.fit(df[cont_features])
    mode_imputer = preprocessing.Imputer(strategy="most_frequent")
    mode_imputer.fit(df[cat_features])
    return mean_imputer, mode_imputer

def impute_missing_data(df, featuers, imputers):
    all_cont_features = get_continuous_columns(df)
    cont_features = []
    cat_features = []
    for feature in features:
        if feature in all_cont_features:
            cont_features.append(feature)
        else:
            cat_features.append(feature)

    df[cont_features] = imputers[0].transform(df[cont_features])
    df[cat_features] = imputers[1].transform(df[cat_features])

 
def get_heat_map_corr(df):
    corr = df.select_dtypes(include = ['number']).corr()
    sns.heatmap(corr, square=True)
    plt.xticks(rotation=70)
    plt.yticks(rotation=70)
    return corr

def get_target_corr(corr, target):
    return corr[target].sort_values(axis=0,ascending=False)

def one_hot_encode(df):
   features = get_categorical_columns(df)
   return pd.get_dummies(df, columns=features)

def rmse(y_orig, y_pred):
    return math.sqrt(metrics.mean_squared_error(y_orig,y_pred))
  
def fit_model(estimator, grid, X_train, y_train):
   grid_estimator = model_selection.GridSearchCV(estimator, grid, scoring = metrics.make_scorer(rmse), cv=10, n_jobs=1)
   grid_estimator.fit(X_train, y_train)
   print(grid_estimator.grid_scores_)
   print(grid_estimator.best_params_)
   print(grid_estimator.best_score_)
   print(grid_estimator.score(X_train, y_train))
   return grid_estimator.best_estimator_

def feature_importances(estimator):
    return estimator.feature_importances_

def predict(estimator, X_test):
    return estimator.predict(X_test)

#to uncover the mismatch of levels between train and test data
def merge(df1, df2):
    return pd.concat([df1, df2])

def split(df, ind):
    return (df[0:ind], df[ind:])

def viz_cont_cont(df, features, target):
    for feature in features:
        sns.jointplot(x = feature, y = target, data = df)
        
def viz_cat_cont_box(df, features, target):
    for feature in features:
        sns.boxplot(x = feature, y = target,  data = df)
        plt.xticks(rotation=45)

def viz_cont(df, features):
    for feature in features:
        sns.distplot(df[feature],kde=False)

os.getcwd
os.environ['PATH'] += os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin\\'
#Connect to DB using connection string
#Write select query
#Put it in a data frame

house_train = pd.read_csv("C:\\Data Science\\data\\House_Prices\\train.csv")
house_train.shape
house_train.info()

house_test = pd.read_csv("C:\\Data Science\\data\\House_Prices\\test.csv")
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
quality_dict = {None: 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
transform_cat_to_cont(house_data, ordinal_features, quality_dict)

print(get_continuous_columns(house_data))
print(get_categorical_columns(house_data))

house_train, house_test = split(house_data, house_train.shape[0])
house_train.shape
house_test.shape

#filter missing data columns
missing_features = get_features_missing_data(house_train, 0)
filter_features(house_train, missing_features)
house_train.shape
house_train.info()

#smooth the sale price using log transformation(smoothening outlier data)
house_train['log_sale_price'] = np.log(house_train['SalePrice'])
features = ['SalePrice','log_sale_price']
viz_cont(house_train, features)

#explore relationship of neighborhood to saleprice
target = 'SalePrice'
features = ['Neighborhood']
viz_cat_cont_box(house_train, features, target)

#explore relationship of livarea and totalbsmt to saleprice
features = ['GrLivArea']
features = ['TotalBsmtSF']
viz_cont_cont(house_train, features, target)

filter_features(house_train, ['Id'])
                               
#explore relation among all continuous features vs saleprice 
corr = get_heat_map_corr(house_train)
get_target_corr(corr, 'SalePrice')
#In case if the data has outliers then some way we have to make them to Bell curve fasion. The same can be done by Log transformation.
get_target_corr(corr, 'log_sale_price')

#do one-hot-encoding for all the categorical features
print(get_categorical_columns(house_train))
house_train1 = one_hot_encode(house_train)
house_train1.shape
house_train1.info()

filter_features(house_train1, ['SalePrice','log_sale_price'])
X_train = house_train1
y_train = house_train['log_sale_price']

#Check the magnitude of Coefficients using Linear Regression
lreg = linear_model.LinearRegression()
lreg.fit(X_train, y_train)
coef = lreg.coef_ 
print(coef)

#Plot Linear Regression Coefficients
predictors = X_train.columns
coef = Series(lreg.coef_, predictors).sort_values()
coef.plot(kind='bar', title='Modal Coefficients with Linear Regression')

#Check the magnitude of Coefficients after L2/Ridge Regression
ridgeReg = linear_model.Ridge(alpha=0.1, normalize=True)
ridgeReg.fit(X_train,y_train)
coef = ridgeReg.coef_ 
print(coef) #Notice that the magnitude of co-efficients got reduced

#Plot L2/Ridge Regression Coefficients
predictors = X_train.columns
coef = Series(ridgeReg.coef_, predictors).sort_values()
coef.plot(kind='bar', title='Modal Coefficients with Ridge Regression')

##Check the magnitude of Coefficients after L1/LASSO Regression
LassoReg = linear_model.Lasso(alpha=0.0001, normalize=True)
LassoReg.fit(X_train,y_train)
coef = LassoReg.coef_ 
print(coef)

#Plot L1/LASSO Regression Coefficients
predictors = X_train.columns
coef = Series(LassoReg.coef_, predictors).sort_values()
coef.plot(kind='bar', title='Modal Coefficients with LASSO Regression')

#####Fit models##########
#Using plain vanilla Linear Regression
lreg_estimator = linear_model.LinearRegression()
lreg_grid = {'normalize':[True]}
lreg_model = fit_model(lreg_estimator, lreg_grid, X_train, y_train)

#L2 regulizer/Ridge Regression
#L2/Ridge Regulizer reduces the magnitue of coefficients
ridge_estimator = linear_model.Ridge()
ridge_grid = {'alpha':[0.01, 0.05, 0.07]}
ridge_model = fit_model(ridge_estimator, ridge_grid, X_train, y_train)

#L1 regulizer/Lasso
#L1/LASSO Regulizer assigns zeros to un-important features
lasso_estimator = linear_model.Lasso()
lasso_grid = {'alpha':[0, 0.1, 0.5, 0.7, 1]}
lasso_model = fit_model(lasso_estimator, lasso_grid, X_train, y_train)

#ElasticNet is the Hybrid approach (L1+L2)
elasticNet_estimator = linear_model.ElasticNet()
elasticNet_grid = {'alpha':[1], 'l1_ratio':[0.5]}
elasticNet_model = fit_model(elasticNet_estimator, elasticNet_grid, X_train, y_train)
#

