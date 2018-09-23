# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 00:24:53 2018

@author: Ankita
"""
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model #You need have GraphViz already installed in your machine

def plot_loss_accuracy(history):
    historydf = pd.DataFrame(history.history, index=history.epoch)
    plt.figure(figsize=(8, 6))
    historydf.plot(ylim=(0, max(1, historydf.values.max())))
    loss = history.history['loss'][-1]
    acc = history.history['acc'][-1]
    plt.title('Loss: %.3f, Accuracy: %.3f' % (loss, acc))
    

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

model = Sequential()
#Hidden Layer1
model.add(Dense(512, input_shape=(784,), activation='sigmoid' ))
model.add(Dense(10,  input_shape=(512,), activation='softmax'))
print(model.summary())

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

#Plot Model
os.getcwd()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# =============================================================================
# #Visualization of network. But takes long time!!!
# from ann_visualizer.visualize import ann_viz;
# os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin'
# ann_viz(model, title = "model_NN.pdf", )
# =============================================================================
epochs = 20
batchsize = 5
history = model.fit(x=X_train, y=y_train, verbose=1, epochs=epochs, batch_size=batchsize, validation_split=0.2)
print(model.get_weights())
plot_loss_accuracy(history)

#Predictions on Test data
digit_test = pd.read_csv("C:\\Data Science\\Deep Learning\\data\\Digit_Recognizer\\test.csv")
digit_test.shape
digit_test.info()

X_test = digit_test.values.astype('float32')/255.0

pred = model.predict_classes(X_test)
submissions= pd.DataFrame({"ImageId": list(range(1,len(pred)+1)), "label": pred})
submissions.to_csv("submission_DigitsRec3.csv", index=False, header=True)
