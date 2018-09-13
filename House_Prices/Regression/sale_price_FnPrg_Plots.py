# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 12:37:04 2018

@author: Ankita
"""
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
from sklearn import model_selection, metrics

#to uncover the mismatch of levels between train and test data
def merge(df1, df2):
    return pd.concat([df1, df2])

#To get all continuous columns
def get_continuous_columns(df):
    return df.select_dtypes(include=['number']).columns

#To get all categorial columns
def get_categorial_columns(df):
    return df.select_dtypes(exclude=['number']).columns

def transform_cont_to_cat(df, features):
    for feature in features:
        df[feature] = df[feature].astype('category')

def transform_cat_to_cont(df, features, mappings):
    for feature in features:
        null_idx = df[feature].isnull()
        df.loc[null_idx, feature] = None
        df[feature] = df[feature].map(mappings)

#To separate train data and train data
def split(df, ind):
    return(df[0:ind],df[ind:])

#To get columns whose data is missing
def get_features_missing_data(df, cutoff):
    total_missing = df.isnull().sum()
   # n = df.shape[0]
    to_delete = total_missing[(total_missing)> cutoff]
    return list(to_delete.index)

#For dropping the columns
def filter_features(df, features):
    df.drop(features, axis = 1, inplace = True)
    
#Visualize preferably Continuous data with count
def viz_cont(df, features):
    for feature in features:
        sns.distplot(df[feature], kde=False)
        
def viz_cont_cont(df, features, target):
    for feature in features:
        sns.jointplot(x= feature, y= target, data= df)
        
def viz_cat_cont_box(df, features, target):
    for feature in features:
        sns.boxplot(x= feature, y= target, data= df)
        plt.xticks(rotation = 45)
        
def get_heat_map_corr(df):
    corr = df.select_dtypes(include = ['number']).corr()
    sns.heatmap(corr, square=True)
    plt.xticks(rotation=70)
    plt.yticks(rotation=70)
    return corr

def get_target_corr(corr, target):
    return corr[target].sort_values(axis =0, ascending= False)

#To get the imputers
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

#To impute the missing data
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
 
#Convert categoric to One hot encoding using get_dummies
def one_hot_encode(df):
   features = get_categorical_columns(df)
   return pd.get_dummies(df, columns=features)

#evaluate using rmse
def rmse(y_orig, y_pred):
    return math.sqrt(metrics.mean_squared_error(y_orig,y_pred))
  
#feature_importances_: Every feature has an importance with a priority number. Now we want to use best estimator along with very very importance features
def feature_importances(estimator):
    return estimator.feature_importances_

#To do model fitting
def fit_model(estimator, grid, X_train, y_train):
   grid_estimator = model_selection.GridSearchCV(estimator, grid, scoring = metrics.make_scorer(rmse), cv=10, n_jobs=10)
   grid_estimator.fit(X_train, y_train)
   print(grid_estimator.grid_scores_)
   print(grid_estimator.best_params_)
   print(grid_estimator.best_score_)
   print(grid_estimator.score(X_train, y_train))
   return grid_estimator.best_estimator_

#To do prediction on model.
def predict(estimator, X_test):
    return estimator.predict(X_test)

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
print(get_categorial_columns(house_data))

#convert numerical columns to categorical type 
features = ['MSSubClass']
transform_cont_to_cat(house_data, features)

#map string categoical values to numbers
ordinal_features = ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageQual','GarageCond','PoolQC']
quality_dict = {'NA': 0, 'Po': 1, 'Fa': 2, 'Ta': 3, 'Gd': 4, 'Ex': 5}
transform_cat_to_cont(house_data, ordinal_features, quality_dict)

print(get_continuous_columns(house_data))
print(get_categorial_columns(house_data))

house_train, house_test = split(house_data, house_train.shape[0])

#filter missing data columns
missing_features = get_features_missing_data(house_train, 0)
filter_features(house_train, missing_features)
house_train.shape
house_train.info()

#smooth the sale price using log transformation(smoothening outlier data)
house_train['log_sale_price'] = np.log(house_train['SalePrice'])
features =['SalePrice','log_sale_price']
viz_cont(house_train, features)

#explore relationship of neighborhood to saleprice
target = 'SalePrice'
features = ['Neighborhood']
viz_cat_cont_box(house_train, features, target)

#explore relationship of livarea and totalbsmt to saleprice
features = ['GrLivArea','TotalBsmtSF']
viz_cont_cont(house_train, features, target)

filter_features(house_train, ['Id'])

#explore relation among all continuous features vs saleprice 
corr = get_heat_map_corr(house_train)
get_target_corr(corr, 'SalePrice')
get_target_corr(corr, 'log_sale_price')

#do one-hot-encoding for all the categorical features
print(get_categorical_columns(house_train))
house_train1 = one_hot_encode(house_train)
house_train1.shape
house_train1.info()

filter_features(house_train1, ['SalePrice','log_sale_price'])
X_train = house_train1
y_train = house_train['log_sale_price']






