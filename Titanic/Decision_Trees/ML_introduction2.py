# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 22:19:04 2018
x
@author: Ankita
"""

import pandas as pd
from sklearn import tree
import os
import io
import pydot

os.environ["PATH"] = os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin'
titanic_train = pd.read_csv("C:\\Data Science\\data\\Titanic_dataset\\train.csv")

#EDA
titanic_train.shape
titanic_train.info()
titanic_train.describe()
#Transformation of non numneric cloumns
#There is an exception with the pclass. Though it's co-incidentally is a number but it's a classification but not a number.
#titanic_train1 = titanic_train[['Pclass', 'Sex', 'Embarked', 'Fare']]

#Convert categoric to One hot encoding using get_dummies
train1 = pd.get_dummies(titanic_train, columns =['Pclass','Sex','Embarked'])
train1.shape
train1.info()
train1.describe
#now the drop non numerical columns where we will not be applying logic. Something like we will not apply logic on names, passengerID ticket id etc...
x_train1 = train1.drop(['PassengerId', 'Name', 'Age','Ticket', 'Cabin', 'Survived'],1)
x_train1.shape
y_train1 = titanic_train['Survived']

#.fit builds the model. In this case the model building is using Decission Treee Algorithm
dt = tree.DecisionTreeClassifier()
dt.fit(x_train1, y_train1)

#visualize the decission tree
dot_data = io.StringIO()
tree.export_graphviz(dt, out_file = dot_data, feature_names = x_train1.columns)
graph = pydot.graph_from_dot_data(dot_data.getvalue()) [0]
graph.write_pdf("DT-AugmentedColumns.pdf")

#predict the outcome using decission tree
titanic_test = pd.read_csv("C:\\Data Science\\data\\Titanic_dataset\\test.csv")
titanic_test.shape
titanic_test.info()#Found that one row has Fare = null in test data. Instead of dropping this column, let's take the mean of it.
titanic_test.Fare[titanic_test['Fare'].isnull()] = titanic_test['Fare'].mean()

#Now apply same get_dummies and drop columns on test data as well like above we did for train data
titanic_test1 = pd.get_dummies(titanic_test, columns = ['Pclass','Sex','Embarked'])
x_test1 = titanic_test1.drop(['PassengerId', 'Name', 'Age','Ticket', 'Cabin'],1)
#Apply the model on Furture/test data
titanic_test['Survived']= dt.predict(x_test1)
titanic_test.to_csv("Submission_Attempt2.csv", columns= ['PassengerId','Survived'],index = False)




