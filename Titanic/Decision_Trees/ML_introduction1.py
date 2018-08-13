# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 07:34:22 2018
 
@author: Ankita
"""
import pandas as pd 
from sklearn import tree
import os
import io
import pydot

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
titanic_train = pd.read_csv("C:\\Data Science\\data\\Titanic_dataset\\train.csv")

#Exploratory Data Analysis(EDA)
titanic_train.shape
titanic_train.info()
titanic_train.describe
x_titanic_train= titanic_train[['Pclass','Parch']]
y_titanic_train= titanic_train[['Survived']]

#build the decision tree model 
dt = tree.DecisionTreeClassifier()
dt.fit(x_titanic_train,y_titanic_train)

#visualize the decision tree
dot_data = io.StringIO()
tree.export_graphviz(dt, out_file = dot_data, feature_names = x_titanic_train.columns)
graph = pydot.graph_from_dot_data(dot_data.getvalue()) [0]
graph.write_pdf("DS-DT.pdf")
