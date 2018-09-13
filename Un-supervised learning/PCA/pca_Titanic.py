# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 11:09:19 2018

@author: Ankita
"""
import os
import pandas as pd
from sklearn import decomposition
import seaborn as sns

os.getcwd()
os.environ['PATH'] = os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin\\'

titanic_train = pd.read_csv("C:\\Data Science\\data\\Titanic_dataset\\train.csv")
#EDA
titanic_train.shape
titanic_train.info()

titanic_train1 = pd.get_dummies(titanic_train, columns=['Pclass', 'Sex', 'Embarked'])
titanic_train1.shape
titanic_train1.info()

X_train = titanic_train1.drop(['PassengerId','Age','Cabin','Ticket', 'Name','Survived'], 1)
X_train.info()

#Here comes the PCA!
pca= decomposition.PCA(n_components =4)
pca.fit(X_train)

print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())

#Transformation of PCA happens here
transformed_X_train = pca.transform(X_train)
y_train = titanic_train['Survived']

#Assign transformed PCA data into new data frame for visualaiztion purpose
transformed_df = pd.DataFrame(data= transformed_X_train,columns = ['pc1', 'pc2', 'pc3', 'pc4'])
#See whethere PC1 and PC2s are orthogonal are not!
sns.jointplot('pc1', 'pc2', transformed_df)