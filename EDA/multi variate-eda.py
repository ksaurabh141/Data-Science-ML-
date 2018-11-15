# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 23:16:09 2018

@author: Ankita
"""
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

os.getcwd()
titanic_train = pd.read_csv('C:\\Data Science\\data\\Titanic_dataset\\train.csv')

#EDA
titanic_train.shape
titanic_train.info()

sns.FacetGrid(titanic_train, row = 'Survived', col= 'Sex').map(sns.countplot, 'Pclass')
sns.FacetGrid(titanic_train, row = 'Survived', col= 'Sex').map(sns.kdeplot, 'Fare')
sns.FacetGrid(titanic_train, row = 'Survived', col= 'Sex').map(sns.kdeplot, 'Age')

sns.FacetGrid(titanic_train, row="Pclass", col="Sex", hue="Survived").map(sns.kdeplot, "Age").add_legend()

sns.FacetGrid(titanic_train, row="Pclass", col="Sex", hue="Survived").map(plt.scatter, "Parch", "SibSp").add_legend()

sns.FacetGrid(titanic_train, row="Survived", col="Sex").map(plt.scatter, "Pclass", "SibSp", "Parch")
