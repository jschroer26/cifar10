
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from tensorflow import keras
from tensorflow.keras.preprocessing import image

IMG_HEIGHT = 150
IMG_WIDTH = 150




img = image.load_img('C:\\Users\\jschro26\\.keras\\datasets\\cats_and_dogs_filtered\\train\\cats\\cat.3.jpg', target_size = (IMG_WIDTH, IMG_HEIGHT))
img1 = image.img_to_array(img)
img2 = np.expand_dims(img1, axis = 0)

pv = model.predict(img2)

print ( 'prediction on cat3.jpg', pv )


model = tf.keras.models.load_model ('epic-cifor10-classifier.h5')
predictions = model.predict([x_test])

#checking that the state is preserved (from tf.org)
new_predictions = model.predict(x_test)
np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)


batch_size=32
epochs = 15
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), verbose=0) 

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
#
plt.save ('ecifor10class.png')




##now we want to see this for the 0th image
print(np.argmax(predictions[0]))

plt.imshow(x_test[0]) #shows the 0th image you can change the image
plt.show()