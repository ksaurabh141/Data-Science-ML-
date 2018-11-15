# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 21:47:42 2018

@author: Ankita
"""
import pandas as pd
import os
from sklearn import preprocessing
from sklearn import naive_bayes
from sklearn import model_selection

os.getcwd()
#changes working directory
os.chdir('C:\\Data Science\\data\\Titanic_dataset\\')

titanic_train = pd.read_csv('train.csv')
titanic_train.info()
titanic_train.shape

titanic_test =pd.read_csv('test.csv')
titanic_test['Survived'] = None
titanic_test.info()
titanic_test.shape

#it gives the same never of levels for all the categorical variables
titanic = pd.concat([titanic_train,titanic_test])
titanic.info()
titanic.shape

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
    if (age >= 10 and age <= 15):
        return 'Child'
    if (age <= 25):
        return 'Young'
    if (age <= 50):
        return 'Middle'
    else:
        return 'Old'

titanic['Age1'] = titanic['Age'].map(convert_age)

titanic['FamilySize'] = titanic['SibSp'] + titanic['Parch'] + 1

def convert_familysize(size):
    if (size == 1):
        return 'Single'
    if (size <= 3):
        return 'Small'
    if (size <= 6):
        return 'Medium'
    else:
        return 'Large'
    
titanic['FamilySize1'] = titanic['FamilySize'].map(convert_familysize)

#convert categorical columns to one-hot encoded columns
titanic1 = pd.get_dummies(titanic, columns = ['Pclass','Sex','Age1','FamilySize1','Embarked','Title'])
titanic1.info()
titanic1.shape

titanic2 = titanic1.drop(['PassengerId','Survived','Name','Age','SibSp','Parch','Cabin','Ticket'],axis = 1, inplace = False)
titanic2.info()
titanic2.shape

X_train = titanic2[0:titanic_train.shape[0]]
X_train.info()
X_train.shape
Y_train = titanic_train['Survived']

#DT doesn't give the probablity of classification. If there are times business may need classification with the probability
#That's where we can use NB
nb_estimator = naive_bayes.GaussianNB()
#nb_estimator = naive_bayes.BernoulliNB()
#nb_estimator = naive_bayes.MultinomialNB()

mean_val_score = model_selection.cross_val_score(nb_estimator, X_train, Y_train,cv = 10).mean()
nb_estimator.fit(X_train,Y_train)

#nb_estimator.class_prior_
#mean
nb_estimator.sigma_
#Deviation
nb_estimator.theta_

X_test = titanic2[titanic_train.shape[0]:]
X_test.info()
X_test.shape

titanic_test['Survived'] = nb_estimator.predict(X_test)
titanic_test.to_csv('submission_GaussianNB.csv', columns=['PassengerId','Survived'],index=False)
#titanic_test.to_csv('submission_BernoulliNB.csv', columns=['PassengerId','Survived'],index=False)
#titanic_test.to_csv('submission_MultinomialNB.csv', columns=['PassengerId','Survived'],index=False)
os.getcwd()
#.predict_prob will work only after .predict
#predict_proba will give the probability based output classification
titanic_test['Survived1'] = nb_estimator.predict_proba(X_test)
titanic_test.to_csv('submission_GaussianNB_prob.csv', columns=['PassengerId','Survived'],index=False)
#titanic_test.to_csv('submission_BernoulliNB_prob.csv', columns=['PassengerId','Survived'],index=False)
#titanic_test.to_csv('submission_MultinomialNB_prob.csv', columns=['PassengerId','Survived'],index=False)

