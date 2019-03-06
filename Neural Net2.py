# -*- coding: utf-8 -*-
"""
@author: znewe_000
"""
# Fashion MNIST dataset classification using tensorflow
# I wanted to try some image classification and see how 
# tensorflow makes it easy.

# Imports
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['Shirt', 'Pants', 'Sweater', 'Dress', 'Coat', 
               'Flops', 'Shirt', 'Shoes', 'Bag', 'Boots']

# The images are all 255 pixels, lets normalize and zoom in
# to get a good idea of the pixeled shape of each piece of 
# clothing
train_images = train_images / 255.0

test_images = test_images / 255.0

# build the model with three layers, I want to flatten the image 
# to an array, then compute rectified linear, then find the max
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Actually build it with tensorflow optimizer, loss, and metrics
# the adam optimizer works well for finding different gradients
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training time
model.fit(train_images, train_labels, epochs=5)

# Test time
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

# Prediction time
predictions = model.predict(test_images)
predictions[0]
np.argmax(predictions[0])
test_labels[0]

# Can also make single predictions by giving single images
img = test_images[0]
img = (np.expand_dims(img,0))
predictions_single = model.predict(img)

print(predictions_single)
