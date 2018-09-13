# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 10:45:23 2018

@author: Ankita
"""
import numpy as np
from sklearn import decomposition 
import pandas as pd
import seaborn as sns

#Highly correlated columns (X1, X2)
df1 = pd.DataFrame({'Age':[10,20,30,40],'Fare':[15,25,35,45]})
sns.jointplot('Age','Fare',df1)
pca= decomposition.PCA(n_components = 1)
pca.fit(df1)
pca.components_[0]

#understand how much variance captured by each principal component
df1_pca= pca.transform(df1)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())
df1_pca = pca.transform(df1)
df1_pca.shape

#Not much correlated
df2 = pd.DataFrame({'Age':[10, 20, 30, 4000],'Fare':[23, 79, 1, 5]})
sns.jointplot('Age','Fare',df2)
pca = decomposition.PCA(n_components=1)
pca.fit(df2)
pca.components_[0]
#understand how much variance captured by each principal component
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())
df2_pca = pca.transform(df2)
print(df2_pca)