# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 22:19:30 2018

@author: Ankita
"""

import pandas as pd
import os
from sklearn import tree
from sklearn import model_selection
#import io
#import pydot #if we need to use any external .exe files.... Here we are using dot.exe

os.environ['PATH'] += os.pathsep + "C:\\Program Files (x86)\\Graphviz2.38\\bin\\"
Titanic_train = pd.read_csv('C:\\Data Science\\data\\Titanic_dataset\\train.csv')
#EDA
Titanic_train.shape
Titanic_train.info()
Titanic_train.describe
#Convert categoric to One hot encoding using get_dummies
Titanic_train1 = pd.get_dummies(Titanic_train, columns=['Pclass','Sex','Embarked'])
Titanic_train1.shape
Titanic_train1.info()
Titanic_train1.describe
#now the drop non numerical columns where we will not be applying logic. Something like we will not apply logic on names, passengerID ticket id etc...
X_train = Titanic_train1.drop(['PassengerId','Name','Age','Ticket','Cabin', 'Survived'],1)
Y_train = Titanic_train['Survived']
X_train.shape
X_train.info()
X_train.describe

dt = tree.DecisionTreeClassifier(random_state = 2018 )
#Build the decision tree model
param_grid = {'max_depth':[15,100], 'min_samples_split':[2,6], 'criterion':['gini','entropy']}
print(type(param_grid))
dt_grid = model_selection.GridSearchCV(dt, param_grid, cv=10, n_jobs=9)
print(type(dt_grid))
#.fit builds the model.
#dt.fit(X_train,Y_train)
dt_grid.fit(X_train,Y_train)
#type(dt)

dt_grid.grid_scores_
dt_grid.best_params_
dt_grid.best_score_
dt_grid.score(X_train, Y_train)

# #visualize the decission tree
#dot_data= io.StringIO()
#tree.export_graphviz(dt, out_file = dot_data, feature_names =X_train.columns)
#graph = pydot.graph_from_dot_data(dot_data.getvalue()) [0]
#graph.write_pdf("DTwithParameters.pdf")

#predict the outcome using decission tree
Titanic_test = pd.read_csv('C:\\Data Science\\data\\Titanic_dataset\\test.csv')
Titanic_test.shape
Titanic_test.info()
Titanic_test.Fare[Titanic_test['Fare'].isnull()] = Titanic_test['Fare'].mean()
#Now apply same get_dummies and drop columns on test data as well like above we did for train data
Titanic_test1 = pd.get_dummies(Titanic_test, columns = ['Pclass','Sex','Embarked'])
X_test1 = Titanic_test1.drop(['PassengerId','Name','Age','Ticket','Cabin'],1)
#Apply the model on Furture/test data
Titanic_test['Survived'] = dt_grid.predict(X_test1)
Titanic_test.to_csv("Submission1.csv", columns=['PassengerId','Survived'], index= False)
    

