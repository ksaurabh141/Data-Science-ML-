# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 18:42:06 2018

@author: Ankita
"""
import os
import pandas as pd
from sklearn import decomposition, preprocessing, tree

#returns current working directory
os.getcwd()
os.environ['PATH'] = os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin\\'

titanic_train = pd.read_csv("C:\\Data Science\\data\\Titanic_dataset\\train.csv")
titanic_test = pd.read_csv("C:\\Data Science\\data\\Titanic_dataset\\test.csv")

#EDA
titanic_train.shape
titanic_train.info()

titanic_train1 = pd.get_dummies(titanic_train, columns=['Pclass', 'Sex', 'Embarked'])
titanic_train1.shape
titanic_train1.info()

X_train = titanic_train1.drop(['PassengerId','Age','Cabin','Ticket', 'Name','Survived'], 1)
X_train.info()

#Here comes the PCA!
Scaler = preprocessing.StandardScaler()
Scaler.fit(X_train)
Scaled_data = Scaler.transform(X_train)
print(X_train)
print(Scaled_data)

pca= decomposition.PCA(n_components= 4)
pca.fit(Scaled_data)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())

#Transformation of PCA happens here
transformed_X_train= pca.transform(Scaled_data)
y_train = titanic_train['Survived']

dt = tree.DecisionTreeClassifier()
dt.fit(transformed_X_train, y_train)


titanic_test.info() #Found that one row has Fare = null in test data. Instead of dropping this column, let's take the mean of it.
titanic_test.Fare[titanic_test['Fare'].isnull()] = titanic_test['Fare'].mean()

#Now apply same get_dummies and drop columns on test data as well like above we did for train data
titanic_test1 = pd.get_dummies(titanic_test,columns=['Pclass', 'Sex', 'Embarked'])
X_titanic_test = titanic_test1.drop(['PassengerId','Age','Cabin','Ticket', 'Name'],1)

pca = decomposition.PCA(n_components=4)
pca.fit(X_titanic_test)

X_transformed_Test = pca.transform(X_titanic_test)

#Apply the model on Furture/test data

titanic_test['Survived'] = dt.predict(X_transformed_Test)
titanic_test.to_csv("Submission_PCA.csv",columns=['PassengerId','Survived'],index=False)
