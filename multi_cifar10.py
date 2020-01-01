#Copyright 2020 Joseph E. Schroer, All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt

from keras.datasets import cifar10 #loads dataset
from keras.utils import np_utils #transforms labels into categories
from keras.models import Sequential 
from keras.layers import Dense, Conv2D, BatchNormalization, Flatten, Dropout, MaxPooling2D
from keras.optimizers import adam 


#load the data
(x_train, y_train), (x_test, y_test) = cifar10.load_data() 

#maybe add show an image.. for fun :)

#define the number of classes
nClasses = 10
cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"] 

y_train = np_utils.to_categorical(y_train,nClasses)
y_test = np_utils.to_categorical(y_test,nClasses)

#Prepare the Pixel Data -->float then> rescaling on range [0,1]
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train/255.0
x_test = x_test/255.0 

print (x_train.shape) #tells us # of images, their size (32x32) and the colors/channels (RGB)
print (y_train.shape) #tells us # of images, # of categories

#now, let's build the keras neural network
def new_model ():
    model = Sequential()
    #keras.constraints.MaxNorm(max_value=2, axis=0)
    model.add(Conv2D(32, (3, 3), padding = 'same', activation = 'relu', input_shape=(32, 32, 3)))# adding a layer of 32 neurons using relu to learn the images
    model.add(Conv2D(32, (3, 3), padding = 'same', activation = 'relu', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D()) #this seems to do the same thing as kernel_constraint seen in layer 3 to avoid overfitting
    model.add(Dropout(0.3))
   
    model.add(Conv2D(128, (3, 3), padding = 'same', activation = 'relu', input_shape=(32, 32, 3)))# adding a layer of 128 neurons using relu to learn the images
    model.add(Conv2D(128, (3, 3), padding = 'same', activation = 'relu', input_shape=(32, 32, 3))) #
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Dropout(0.4))
    
    model.add(Flatten()) #flatten the images before final level to help processing
    from keras.constraints import max_norm
    model.add(Dense(512, kernel_constraint=max_norm(2.), activation='relu')) # adding a layer of 512 neurons using relu to learn the images****
    model.add(Dense(512, kernel_constraint=max_norm(2.), activation='relu')) 
    model.add(Dropout(0.5))
    model.add(Dense(nClasses, activation='softmax'))    # adding a final layer of 10 neurons using softmax to learn the images
    return model

#define the model
model = new_model()
model.summary()
#fit the model
model.compile(optimizer='adam', #could use stochaistic gradient descent!
			loss='categorical_crossentropy', #do not use sparse_categorical_crossentropy as it expects a certain dense(1) one neuron classification
			metrics=['accuracy']) #remember the idea is to minimize loss not increase accuracy

#Here we are telling the model how many times to run (epochs)
batch_size=64
epochs = 50
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), verbose=0) 

#save the model
model.save('epic-cifor10-classifier3.h5')
print ('model is saved')

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

##Ploting 
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
plt.savefig ('cifor10_3.png')
print ('Figure is saved. Have a nice day! :)')


    
