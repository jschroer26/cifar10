from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import cifar10 #loads dataset

from keras.utils import np_utils #transforms labels into categories

from tensorflow import keras
from tensorflow.keras.preprocessing import image


model = tf.keras.models.load_model ('epic-cifor10-classifier.h5')#make sure you check the file name

#load the data defining x,y test & train
(x_train, y_train), (x_test, y_test) = cifar10.load_data() 
#define the number of classes
nClasses = 10
cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"] 

y_train = np_utils.to_categorical(y_train,nClasses)
y_test = np_utils.to_categorical(y_test,nClasses)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train/255.0
x_test = x_test/255.0 

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
