# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 16:26:05 2018

@author: Ankita
"""

import os
import pandas as pd
from sklearn.externals import joblib

os.environ['PATH'] += os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin\\'

#predict the outcome using decision tree
titanic_test = pd.read_csv("C:\\Data Science\\data\\Titanic_dataset\\test.csv")
titanic_test.Fare[titanic_test['Fare'].isnull()] = titanic_test['Fare'].mean()

titanic_test1 = pd.get_dummies(titanic_test, columns=['Pclass', 'Sex', 'Embarked'])
X_test = titanic_test1.drop(['PassengerId','Age','Cabin','Ticket', 'Name'], 1)

#Use load method to load Pickle file
dtree = joblib.load("TitanicVer1.pkl")
titanic_test['Survived'] = dtree.predict(X_test)
titanic_test.to_csv("submissionUsingJobLib.csv", columns=['PassengerId','Survived'], index=False)

