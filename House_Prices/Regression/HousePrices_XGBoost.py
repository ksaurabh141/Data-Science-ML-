# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 10:35:02 2018

@author: Ankita
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import xgboost as xgb #GBM algorithm
from xgboost import XGBRegressor

os.getcwd()
os.environ['PATH'] = os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin\\'

train_data = pd.read_csv("C:\\Data Science\\data\\House_Prices\\train.csv")
test_data = pd.read_csv("C:\\Data Science\\data\\House_Prices\\test.csv")

#Number columns are picked and Id and SalePrice are removed from the list.
previous_num_columns = train_data.select_dtypes(exclude=['object']).columns.values.tolist()
previous_num_columns.remove('Id')
previous_num_columns.remove('SalePrice')
print(previous_num_columns)

#Delete Outlier Data
#Check the MasVnrArea,LotFrontage,LotArea columns trend in Train and Test -- Any column can be viewed
test_column = 'LotArea'
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
sns.kdeplot(train_data[test_column], ax=ax1)
sns.kdeplot(test_data[test_column], ax=ax2,color="r")

print(train_data.shape)
train_data.drop(train_data[train_data["MasVnrArea"] > 1500].index, inplace=True)
train_data.drop(train_data[train_data["LotFrontage"] > 200].index, inplace=True)
train_data.drop(train_data[train_data["LotArea"] > 60000].index, inplace=True)
print(train_data.shape)

#The record count is stored -- To separate Train and Test from Combined Data
train_length = train_data.shape[0]
#Both Train and Test are combined to perform EDA and FE
combined_data = pd.concat([train_data.loc[:, : 'SalePrice'], test_data])
combined_data = combined_data[test_data.columns]
print(combined_data.shape)
#Filling missing Values
#missing data columns are obtained -- 
has_null_columns= combined_data.columns[combined_data.isnull().any()].tolist()
#A function to fill the missing data with given value
def fill_missing_combined_data(column, value):
    combined_data.loc[combined_data[column].isnull(),column] = value
    
#Lot Frontage -- Filled with median of the Neighborhood, grouped
lf_neighbor_map = combined_data['LotFrontage'].groupby(combined_data["Neighborhood"]).median()
print(lf_neighbor_map)
#Get the records with missing values and fill median of Neighborhood group.
rows = combined_data['LotFrontage'].isnull()
combined_data['LotFrontage'][rows] = combined_data['Neighborhood'][rows].map(lambda neighbor : lf_neighbor_map[neighbor])

#Alley -- All the missing values are filled with NA, which means No Alley
combined_data.shape, combined_data[combined_data['Alley'].isnull()].shape
fill_missing_combined_data('Alley', 'NA')
#FireplaceQu - For Fireplaces 0, FireplaceQu is set to NA, indicating No Fireplace, which is the case of missing 1420 records of data
fill_missing_combined_data('FireplaceQu', 'NA')
fill_missing_combined_data('PoolQC', 'NA')
fill_missing_combined_data('MiscFeature', 'NA')
fill_missing_combined_data('Fence', 'NA')

#MasVnrType filled with None and MasVnrArea with 0
combined_data['MasVnrType'].fillna('None', inplace=True)
combined_data['MasVnrArea'].fillna(0, inplace=True)

#BAsement columns -- BsmtQual / BsmtCond / BsmtExposure / BsmtFinType1 / BsmtFinType2
basement_cols=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1','BsmtFinSF2']

#The data for the missing string type is filled with NA, which means No Basement
for column in basement_cols:
    if 'FinSF'not in column:
        fill_missing_combined_data(column, 'NA')

#Basement related continous columns are filled with 0 -- which means no basement
fill_missing_combined_data('BsmtFinSF1', 0)
fill_missing_combined_data('BsmtFinSF2', 0)
fill_missing_combined_data('BsmtUnfSF', 0)
fill_missing_combined_data('TotalBsmtSF', 0)
fill_missing_combined_data('BsmtFullBath', 0)
fill_missing_combined_data('BsmtHalfBath', 0)

#Garage Columns
garage_cols=['GarageType','GarageQual','GarageCond','GarageYrBlt','GarageFinish','GarageCars','GarageArea']

#The data for the missing string type is filled with NA, which means No Garage
for column in garage_cols:
    if column != 'GarageCars' and column != 'GarageArea':
        fill_missing_combined_data(column, 'NA')
    else:
        fill_missing_combined_data(column, 0)
       
#Fill with Mode
#Electrical - Missing a piece of data, filled with the highest number of occurrences.
sns.countplot(combined_data['Electrical']) #To get the most frequent value which is Mode
fill_missing_combined_data('Electrical', 'SBrkr')

fill_missing_combined_data('MSZoning', 'RL')
fill_missing_combined_data('Utilities', 'AllPub')
fill_missing_combined_data('Exterior1st', 'VinylSd')
fill_missing_combined_data('Exterior2nd', 'VinylSd')
fill_missing_combined_data('KitchenQual', 'TA')
fill_missing_combined_data('SaleType', 'WD')
fill_missing_combined_data('Functional', 'Typ')

#Check if there is any missing data -- It should be empty now
has_null_columns = combined_data.columns[combined_data.isnull().any()].tolist()
print(has_null_columns)

#Feature Engineering

built_year_data['GarageYrBlt'] = built_year_data['GarageYrBlt'].map(lambda g : int(g))
built_year_data['GarageYrBlt'].corr(built_year_data['YearBuilt'])

#You can see that there is a high correlation between YearBuilt and GarageYrBlt.
from sklearn.linear_model import LinearRegression
regr = LinearRegression()
garage_year = built_year_data.loc[:,'GarageYrBlt'].values
built_year = built_year_data.loc[:,'YearBuilt'].values

length = garage_year.shape[0]
garage_year = garage_year.reshape(length, 1)
built_year = built_year.reshape(length, 1)

# Train the model using the training sets
regr.fit(built_year, garage_year)
plt.scatter(built_year, garage_year,  color='blue')
plt.plot(built_year, regr.predict(built_year), color='red',linewidth=3)

combined_data['GarageYrBlt'] = combined_data.apply(lambda row : int(regr.predict(row['YearBuilt']))if row['GarageYrBlt'] == 'NA' else int(row['GarageYrBlt']),axis=1)
#combined_data = sqlContext.createDataFrame(combined_data)
#combined_data.registerTempTable('tmp')
#data = sqlContext.sql('select GarageYrBlt from tmp where Id == 40').show()
# =============================================================================

#YearBuilt and YearRemodAdd determines whether the renovation
#How many years has remoded from built
combined_data['RemodYears'] = combined_data['YearRemodAdd'] - combined_data['YearBuilt']
#Did a remodeling happened from built?
combined_data["HasRemodeled"] = (combined_data["YearRemodAdd"] != combined_data["YearBuilt"]) * 1
#Did a remodeling happen in the year the house was sold?
combined_data["HasRecentRemodel"] = (combined_data["YearRemodAdd"] == combined_data["YrSold"]) * 1
#How many years garage is built?
combined_data['GarageBltYears'] = combined_data['GarageYrBlt'] - combined_data['YearBuilt']

#How many years has build now?
combined_data['Now_YearBuilt'] = pd.Timestamp.now().year - combined_data['YearBuilt']
combined_data['Now_YearRemodAdd'] = pd.Timestamp.now().year - combined_data['YearRemodAdd']
combined_data['Now_GarageYrBlt'] = pd.Timestamp.now().year - combined_data['GarageYrBlt']

#Convert MSSubClass to new categorical column with new value
mssubclass_dict = {20: 'SC20',30: 'SC30',40: 'SC40',45: 'SC45',50: 'SC50',60: 'SC60',70: 'SC70',75: 'SC75',80: 'SC80',85: 'SC85',90: 'SC90',120: 'SC120',150: 'SC150',160: 'SC160',180: 'SC180',190: 'SC190'}
combined_data['MSSubClass'] = combined_data['MSSubClass'].replace(mssubclass_dict)

# =============================================================================
#Encoding attributes that are large and small
good_level_map = {'Street': {'Grvl': 0, 'Pave': 1},
     'Alley': {'NA':0, 'Grvl': 1, 'Pave': 2},
     'Utilities': {'AllPub':3, 'NoSeWa': 1, 'NoSewr': 2, 'ELO': 0},
     'ExterQual': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1,'Po': 0},
     'ExterCond': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1,'Po': 0},
     'BsmtExposure': {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1,'NA': 0},
     'BsmtQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2,'Po': 1,'NA': 0},
     'BsmtCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2,'Po': 1,'NA': 0},
     'BsmtFinType1': {'NA':0, 'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6},
     'BsmtFinType2': {'NA':0, 'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6},
     'HeatingQC': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1,'Po': 0},
     'CentralAir': {'N':0, 'Y':1},
     'KitchenQual': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
     'Functional': {'Typ':0,'Min1':1,'Min2':1,'Mod':2,'Maj1':3,'Maj2':4,'Sev':5,'Sal': 6},
     'FireplaceQu': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0},
     'GarageQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0},
     'GarageCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0},
     'PoolQC': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'NA': 0},
     'Fence': {'GdPrv': 2, 'GdWo': 2, 'MnPrv': 1, 'MnWw': 1, 'NA': 0}
    }
print(combined_data.shape)
print(good_level_map.keys())
keys = list(good_level_map.keys())

#Replace with the values from above dictionary
good_level_data=combined_data[keys].replace(good_level_map)
good_level_data.columns = good_level_data.columns.map(lambda m : m + '_New')
combined_data[good_level_data.columns] = good_level_data[good_level_data.columns]
print(combined_data.shape)
# =============================================================================

#Create new features
str_columns = combined_data.select_dtypes(include=['object']).columns.values
num_columns = combined_data.select_dtypes(exclude=['object']).columns.values

str_columns

#1.Create some boolean features
sns.countplot(combined_data["LotShape"])
# IR2 and IR3 don't appear that often, so just make a distinction between regular and irregular in case of Lot Shape
combined_data["IsRegularLotShape"] = (combined_data["LotShape"] == "Reg") * 1

# Bnk, Low, HLS don't appear that often, so just make a distinction between leveled or not
combined_data["IsLandContourLvl"] = (combined_data["LandContour"] == "Lvl") * 1

# The only interesting "misc. feature" is the presence of a shed.
combined_data["HasShed"] = (combined_data["MiscFeature"] == "Shed") * 1

#Was this house sold in the year it was built?
combined_data["IsSoldInBuiltYear"] = (combined_data["YearBuilt"] == combined_data["YrSold"]) * 1

#2.Simplifications of existing features -- Convert them to good, average and bad
combined_data["SimplOverallQual"] = combined_data.OverallQual.replace(
                                                            {1 : 1, 2 : 1, 3 : 1, # bad
                                                             4 : 2, 5 : 2, 6 : 2, # average
                                                             7 : 3, 8 : 3, 9 : 3, 10 : 3 # good
                                                            })
combined_data["SimplOverallCond"] = combined_data.OverallCond.replace(
                                                            {1 : 1, 2 : 1, 3 : 1, # bad
                                                             4 : 2, 5 : 2, 6 : 2, # average
                                                             7 : 3, 8 : 3, 9 : 3, 10 : 3 # good
                                                             })
    
# 3.Combinations of existing features
# Overall quality of the house
combined_data["OverallGrade"] = combined_data["OverallQual"] * combined_data["OverallCond"]
# Total number of bathrooms
combined_data["TotalBath"] = combined_data["BsmtFullBath"] + (0.5 * combined_data["BsmtHalfBath"]) + combined_data["FullBath"] + (0.5 * combined_data["HalfBath"])
# Total yard area in square feet
combined_data["TotalPorchSF"] = combined_data["OpenPorchSF"] + combined_data["EnclosedPorch"] + combined_data["3SsnPorch"] + combined_data["ScreenPorch"]
# Total SF for house (living, basement, porch, pool)
combined_data["AllSF"] = combined_data["GrLivArea"] + combined_data["TotalBsmtSF"] + combined_data["TotalPorchSF"] + combined_data["WoodDeckSF"] + combined_data["PoolArea"]

#Split Train and Test from Combined Data
train_data_new = combined_data.iloc[:train_length,:]
test_data_new = combined_data.iloc[train_length:, 1:]

#Setting X-Train and Y-Train
train_Y = train_data['SalePrice']
train_X = train_data_new.select_dtypes(exclude=['object']).drop(['Id'], axis=1)

xgb_regressor = XGBRegressor(seed=10)
xgb_regressor.fit(train_X, train_Y)

#Check the Feature Importance and plot the same
feature_importances = pd.Series(xgb_regressor.feature_importances_, train_X.columns.values)
feature_importances = feature_importances.sort_values(ascending=False)
feature_importances.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')

str_columns = combined_data.select_dtypes(include=['object']).columns.values
num_columns = combined_data.select_dtypes(exclude=['object']).columns.values

#Get Continous columns and convert them into Log values
num_columns
combined_data[num_columns] = np.log1p(combined_data[num_columns])

#Onehot Encoding is done for the categorical columns and then delete the original
str_columns
dummies_data = pd.get_dummies(combined_data[str_columns])
combined_data[dummies_data.columns] = dummies_data[dummies_data.columns]
combined_data.drop(str_columns, axis=1, inplace=True)

train_X = combined_data.iloc[:train_length, 1:]
train_Y = train_data['SalePrice']
train_Id = combined_data.iloc[:train_length, 0]

test_X = combined_data.iloc[train_length:, 1:]
test_Id = combined_data.iloc[train_length:, 0]

#Price Comparision for Original Sale Price and log of Sale Price
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))
axis1.hist(train_Y)
train_Y = np.log1p(train_Y)
axis2.hist(train_Y)

# formatting DMatrix to train xgb
dtrain = xgb.DMatrix(train_X, label=train_Y)

# The error metric: RMSE on the log of the sale prices.
from sklearn.metrics import mean_squared_error
#UDF to calculate RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

#UDF to fit the model
def model_fit(xgb_regressor, train_x, train_y, performCV=True, 
              printFeatureImportance=True, cv_folds=5):
    
    # Perform cross-validation
    if performCV:
        xgb_param = xgb_regressor.get_xgb_params()
        cvresult = xgb.cv(xgb_param, dtrain, num_boost_round=xgb_param['n_estimators'], 
                              nfold=cv_folds, metrics='rmse', 
                              early_stopping_rounds=50)
        round_count = cvresult.shape[0]
        mean_rmse = cvresult.loc[round_count-11:round_count-1,'test-rmse-mean'].mean()
        std_rmse = cvresult.loc[round_count-11:round_count-1,'test-rmse-std'].mean()
        
        print("CV RMSE : Mean = %.7g | Std = %.7g" % (mean_rmse, std_rmse))
        
    # fir the train data
    xgb_regressor.fit(train_x, train_y)
    
    # Predict training set
    train_predictions = xgb_regressor.predict(train_x)
    mse = rmse(train_y, train_predictions)
    print("Train RMSE: %.7f" % mse).....
    
    # Print Feature Importance
    if printFeatureImportance:
        feature_importances = pd.Series(xgb_regressor.feature_importances_, train_x.columns.values)
        feature_importances = feature_importances.sort_values(ascending=False)
        feature_importances= feature_importances.head(40)
        feature_importances.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
    
    return xgb_regressor, feature_importances

#Using XGB Regressor algorithm for tuning the model.
xgb_regressor = XGBRegressor(seed=10)
xgb_regressor, feature_importances = model_fit(xgb_regressor,train_X, train_Y)

#Try increasing the learning rate and decreasing the n_estimators Or Vice versa
param_test = {'learning_rate': np.arange(0.2, 0.3, 0.02),
              'n_estimators': np.arange(150, 201, 50),
              'max_depth': np.arange(15, 21, 5),
              'reg_lasso':np.arange(0.55, 0.65, 0.01),
              'reg_ridge':np.arange(0.45, 0.6, 0.01)}

xgb_regressor, feature_importances = model_fit(xgb_regressor,train_X, train_Y)

xgb_predictions = xgb_regressor.predict(test_X)
xgb_predictions = np.expm1(xgb_predictions)

submission = pd.DataFrame({
        "Id": test_Id,
        "SalePrice": xgb_predictions
    })
submission.to_csv("Submission_xgb.csv", index=False)

