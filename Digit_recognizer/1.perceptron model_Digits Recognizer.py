# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 23:07:08 2018

@author: Ankita
"""
from keras import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import pandas as pd
import numpy as np
import os

os.getcwd()
os.environ['PATH'] += os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin'
np.random.seed(100)

digit_train = pd.read_csv("C:\\Data Science\\Deep Learning\\data\\Digit_Recognizer\\train.csv")
digit_train.shape
digit_train.info()

#iloc[:, 1:] Means first to last row and 2nd column to last column
#255.0 --> Convert my data to 255 pixels
X_train = digit_train.iloc[:,1:]/255.0
#X_train.to_csv('aaaaa.csv',index =False)
y_train = np_utils.to_categorical(digit_train["label"])
print(y_train)

y_train.shape


#Here comes the basic Neural network
model = Sequential()
model.add(Dense(10, input_shape=(784,), activation ='softmax'))
print(model.summary())

#mean_squared_error for regression
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x= X_train, y= y_train, verbose = 1,epochs =2, batch_size =2, validation_split = 0.2)
print(model.get_weights())