# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 17:45:24 2018

@author: Ankita
"""
import os
import pandas as pd
from sklearn import ensemble  #This is what we introduced here.
from sklearn import tree
from sklearn import model_selection
import pydot
import io

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

#cv accuracy for bagged tree ensemble
dt_estimator = tree.DecisionTreeClassifier()
#Appy ensemble.BaggingClassificatier
#Base_Estimator = dt_estimator, n_estimators = 5(no. of trees)
#bag_tree_estimator1 = ensemble.BaggingClassifier(base_estimator = dt_estimator, n_estimators =4)
bag_tree_estimator1 = ensemble.BaggingClassifier(base_estimator = dt_estimator, n_estimators =4, max_features = 11)
scores = model_selection.cross_val_score(bag_tree_estimator1, X_train, Y_train, cv=10)
#print(scores)
#print(scores.mean())
bag_tree_estimator1.fit(X_train,Y_train)

#Alternative way with parameters and use GridSearchCV instead of cross_val_score
bag_tree_estimator2 = ensemble.BaggingClassifier(bag_estimator = dt_estimator, n_estimators = 4, random_state = 2017)
bag_grid = {'criterion':['entropy','gini']}
bag_tree_estimator2 = model_selection.GridSearchCV(bag_tree_estimator2, bag_grid, n_jobs =10)
bag_tree_estimator2.fit(X_train, Y_train)

#extracting all the trees build by random forest algorithm
n_tree = 0
for est in bag_tree_estimator1.estimators_:
    dot_data = io.StringIO()
 #   tmp= est.tree_
    tree.export_graphviz(est, out_file = dot_data, feature_names = X_train.columns)
    graph= pydot.graph_from_dot_data(dot_data.getvalue())[0]
    graph.write_pdf("bagtree" + str(n_tree) + ".pdf")
    n_tree = n_tree + 1







