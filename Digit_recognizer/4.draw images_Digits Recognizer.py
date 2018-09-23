# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 00:39:35 2018

@author: Ankita
"""
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

os.getcwd()
os.environ['PATH'] += os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin'
np.random.seed(100)

digit_train = pd.read_csv("C:\\Data Science\\Deep Learning\\data\\Digit_Recognizer\\train.csv")
digit_train.shape
digit_train.info()

#iloc[:, 1:] Means first to last row and 2nd column to last column
#255.0 --> Convert my data to 255 pixels
X_train = digit_train.iloc[:,1:]/255.0
train_features_images=X_train.values.reshape(X_train.shape[0],28,28)
labels = digit_train["label"].values

def show_images(features_images,labels,start, howmany):
    for i in range(start, start+howmany):
        plt.figure(i)
        plt.imshow(features_images[i], cmap=plt.get_cmap('gray'))
        plt.title(labels[i])
    plt.show()
    
show_images(train_features_images, labels, 0, 10)