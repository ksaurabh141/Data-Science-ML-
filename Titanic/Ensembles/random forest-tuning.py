# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 22:35:08 2018

@author: Ankita
"""
import os
import pandas as pd
from sklearn import ensemble
from sklearn import model_selection
#returns current working directory 
os.getcwd()

os.environ['PATH'] += os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin\\'
titanic_train = pd.read_csv("C:\\Data Science\\data\\Titanic_dataset\\train.csv")

#EDA
titanic_train.shape
titanic_train.info()

titanic_train1 = pd.get_dummies(titanic_train, columns=['Pclass', 'Sex', 'Embarked'])
titanic_train1.shape
titanic_train1.info()
titanic_train1.describe

X_train = titanic_train1.drop(['PassengerId','Age','Cabin','Ticket', 'Name','Survived'], 1)
Y_train = titanic_train1['Survived']

#Random Forest classifier
#Remember RandomForest works for Decision Trees only and there is NO Base_Estimator parameter exists
rf_estimator = ensemble.RandomForestClassifier(random_state = 2017)
#n_estimators: no.of trees to be built
#max_features: Maximum no. of features to try with
rf_grid = {'n_estimators' :list(range(50,151,50)),'max_features': [3,6,9]}
rf_grid_estimator = model_selection.GridSearchCV(rf_estimator, rf_grid, cv=10, n_jobs = 10 )
rf_grid_estimator.fit(X_train, Y_train)
rf_grid_estimator.grid_scores_
rf_grid_estimator.best_estimator_
rf_grid_estimator.best_score_
#rf_grid_estimator.best_estimator_.feature_importances_
rf_grid_estimator.score(X_train, Y_train)

titanic_test = pd.read_csv("C:\\Data Science\\data\\Titanic_dataset\\test.csv")
titanic_test.shape
titanic_test.Fare[titanic_test['Fare'].isnull()] = titanic_test['Fare'].mean()

titanic_test1 = pd.get_dummies(titanic_test, columns=['Pclass', 'Sex', 'Embarked'])
titanic_test1.shape
titanic_test1.info()

X_test = titanic_test1.drop(['PassengerId','Age','Cabin','Ticket', 'Name'], 1)
titanic_test['Survived'] = rf_grid_estimator.predict(X_test)
titanic_test.to_csv("submission_rf.csv", columns=['PassengerId','Survived'], index=False)
