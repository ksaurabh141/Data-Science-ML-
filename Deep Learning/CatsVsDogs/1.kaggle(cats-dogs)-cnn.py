# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 21:49:28 2018

@author: Ankita
"""
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
#CNN based models require Conv2D, MaxPooling2D
from keras.layers import Conv2D,MaxPooling2D
#FFNN requires Flatten, Dense 
from keras.layers import Flatten,Dense
from keras import backend as k
import os
#import pandas as pd
os.getcwd()
os.chdir('C:\\Data Science\\Deep Learning\\Practice\\CatsVsDogs')
import utils
#Early stopping is required when system realizes thats there is no improvement after certain epochs
from keras.callbacks import ModelCheckpoint, EarlyStopping
#Pip install pillow
import PIL.Image    
#Prepare small/full data set by calling Utils class method
train_dir, validation_dir, test_dir, nb_train_samples, nb_validation_samples, nb_test_samples = \
            utils.prepare_small_dataset_for_flow(train_dir_original='C:\\Data Science\\Deep Learning\\data\\Dogs Vs Cats Redux\\train', 
                                            test_dir_original='C:\\Data Science\\Deep Learning\\data\\Dogs Vs Cats Redux\\test',
                                            target_base_dir='C:\\Data Science\\Deep Learning\\data\\Dogs Vs Cats Redux\\target base dir')

#Convert all images to standard width and height
img_width, img_height = 150,150
epochs = 1 
batch_size = 20

#Channels first for NON Tensorflow
if k.image_data_format()== 'channels_first':
    input_shape =(3, img_width, img_height)
else:     #For Tensorflow the imput shape is different
    input_shape = (img_width, img_height, 3)
    
#Model begins here
model = Sequential()
#CNN model with 32 filter and the filter size of 3X3 and stride as 1 and padding as 0
model.add(Conv2D(32,(3,3), activation = 'relu', input_shape= input_shape))
model.add(MaxPooling2D(2,2))

#2nd level CNN
model.add(Conv2D(64,(3,3), activation= 'relu'))
model.add(MaxPooling2D((2,2)))

#3rd level CNN
model.add(Conv2D(128,(3,3), activation= 'relu'))
model.add(MaxPooling2D((2,2)))

#4th level of CNN
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
#So far, features got extracted by using CNN and Max Pooling

#Next apply FFNN to classify
model.add(Dense(512, activation='relu'))
#2 outputs (Cats Vs Daogs)
#softmax normalizes the probability Bcoz in the Sigmoid the probability may NOT become Zero.
#Softmax ensures the sum(probablity) must be Zero.
#For example: Image1 one outcome might have came as .7 probability of cat and .2 as dog. And the sum is not = 1.
#Hence Softmax normalizes and make the total probability to Zero
model.add(Dense(2, activation = 'softmax'))
print(model.summary())

model.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])

#Scaling of images from 0-255. Reduse image intensity
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

#If there is no change in continuous 3 epoch(patience=3: Be patient for 3 epochs), then stop early. Heuristically 3-5 is ideal.
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')   
#Save weights in model.h5
save_weights = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)
os.getcwd()
history = model.fit_generator(train_generator, steps_per_epoch=2000//batch_size, epochs=1, validation_data=validation_generator, 
    validation_steps=1000//batch_size,
    callbacks=[early_stopping, save_weights])


