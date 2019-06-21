# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 23:11:58 2019

@author: Priyam
"""
from keras import layers
from keras import models

#Building our CNN
model = models.Sequential()
model.add(layers.Conv2D(32,(5,5),activation='relu',input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(10, activation='softmax'))  #We have to classify 10 layers

model.summary() #Our model is built now. 


#Now, Training and evaluation of the model
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#Adding another dimension coz we need 3 dimensional images and this is black and white so 3rd dimension will be 1
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype(np.float32) / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype(np.float32) / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(train_images, train_labels,
          batch_size=100,
          epochs=5,
          verbose=1)

test_loss, test_acc = model.evaluate(test_images, test_labels)

#Checking a random sample from test dataset
index=0
print('Actual value:{}'.format(np.argmax(test_labels[index])))
test_sample=np.expand_dims(test_images[index],axis=0)
print('Predicted value:{}'.format(np.argmax(model.predict(test_sample))))