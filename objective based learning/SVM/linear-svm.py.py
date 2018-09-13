# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 12:52:43 2018

@author: Ankita
"""
import os
import pandas as pd
from sklearn import preprocessing
from sklearn import svm
from sklearn import model_selection

os.getcwd()
os.environ['PATH'] = os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin\\'

titanic_train = pd.read_csv("C:\\Data Science\\data\\Titanic_dataset\\train.csv")
titanic_train.info()
titanic_train.shape
titanic_test = pd.read_csv("C:\\Data Science\\data\\Titanic_dataset\\test.csv")
titanic_test.info()
titanic_test.shape
titanic_test['Survived'] = None

#it gives the same never of levels for all the categorical variables
titanic = pd.concat([titanic_train,titanic_test])
#create title column from name
def extract_title(name):
     return name.split(',')[1].split('.')[0].strip()
titanic['Title'] = titanic['Name'].map(extract_title)

#create an instance of Imputer class with required arguments
mean_imputer = preprocessing.Imputer()
#compute mean of age and fare respectively
mean_imputer.fit(titanic_train[['Age','Fare']])
#fill up the missing data with the computed means 
titanic[['Age','Fare']] = mean_imputer.transform(titanic[['Age','Fare']])

#create categorical age column from age
def convert_age(age):
    if(age >= 0 and age <= 10): 
        return 'Child'
    elif(age <= 25): 
        return 'Young'
    elif(age <= 50): 
        return 'Middle'
    else: 
        return 'Old'
titanic['Age1'] = titanic['Age'].map(convert_age)

titanic['FamilySize'] = titanic['SibSp'] +  titanic['Parch'] + 1
def convert_familysize(size):
    if(size == 1): 
        return 'Single'
    elif(size <=3): 
        return 'Small'
    elif(size <= 6): 
        return 'Medium'
    else: 
        return 'Large'
titanic['FamilySize1'] = titanic['FamilySize'].map(convert_familysize)

#convert categorical columns to one-hot encoded columns
titanic1 = pd.get_dummies(titanic, columns=['Sex','Pclass','Embarked', 'Age1', 'Title', 'FamilySize1'])
titanic1.shape
titanic1.info()

titanic2 = titanic1.drop(['PassengerId','Name','Age','Ticket','Cabin','Survived'], axis=1, inplace=False)
titanic2.shape

X_train = titanic2[0:titanic_train.shape[0]]
X_train.shape
X_train.info()
y_train = titanic_train['Survived']

lsvm_estimator = svm.LinearSVC(random_state=2017) #Linear Support Vector Classification
lsvm_grid = {'C':[0.1,0.2,0.5,1] } #C: SVM regularization parameter
grid_lsvm_estimator = model_selection.GridSearchCV(lsvm_estimator, lsvm_grid, cv=10, n_jobs=1)
grid_lsvm_estimator.fit(X_train, y_train)
print(grid_lsvm_estimator.grid_scores_)
print(grid_lsvm_estimator.best_score_)
print(grid_lsvm_estimator.best_params_)
print(grid_lsvm_estimator.score(X_train, y_train))
final_model = grid_lsvm_estimator.best_estimator_
final_model.coef_
final_model.intercept_

X_test = titanic2[titanic_train.shape[0]:]
X_test.shape
X_test.info()
titanic_test['Survived'] = grid_lsvm_estimator.predict(X_test)

titanic_test.to_csv('submission_lSVM.csv', columns=['PassengerId','Survived'],index=False)




