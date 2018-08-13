# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 12:29:03 2018

@author: Ankita
"""

import pandas as pd
titanic_train= pd.read_csv("C:\\Data Science\\data\\Titanic_dataset\\train.csv")

print(type(titanic_train))

#explore the dataframe
titanic_train.shape
titanic_train.info()
titanic_train.describe

#access column/columns of a dataframe
titanic_train['Sex']
#or
titanic_train.Sex
titanic_train.Fare
titanic_train[['Survived','Fare']]

#access rows of a data frame
titanic_train.iloc[10]
#Or
titanic_train.loc[10]
#error
titanic_train[3]

titanic_train[0:3]
#Or
titanic_train.iloc[0:3]
#total 4 records will be there.
titanic_train.loc[0:3]

titanic_train[880:889]
#Or
titanic_train.iloc[880:889]
#total 10 records will be there
titanic_train.loc[880:889]

#Get me top n records
titanic_train.head(6)
#Get me bottom n records
titanic_train.tail(6)

#access both rows and columns of a dataframe
titanic_train.iloc[0:3,4]
titanic_train.loc[0:3,4]  #error(cannot do label indexing on <class 'pandas.core.indexes.base.Index'>)
titanic_train.iloc[0:3,4:6]
titanic_train.loc[0:3,4:6] #error(cannot do slice indexing on <class 'pandas.core.indexes.base.Index'>)

#If you wanted to access by column name then use .loc
titanic_train.loc[0:3,'Name']

#conditional access of datafram
titanic_train.loc[titanic_train.Sex == 'Female','Sex']
titanic_train.loc[titanic_train.Sex == 'Female','Name']

#grouping data in data frames
titanic_train.groupby(['Pclass']).size()
titanic_train.groupby(['Pclass','Sex']).size()
titanic_train.groupby(['Pclass']).mean()



