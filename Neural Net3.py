# -*- coding: utf-8 -*-
"""
@author: znewe_000
"""
# Iris dataset
# Simply Python for reference

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

tf.reset_default_graph() 
raw_data =  load_iris()
X_train, X_test, Y_train, Y_test = train_test_split(raw_data.data, raw_data.target, test_size=0.33, random_state=42, stratify= raw_data.target)
X_scaler = MinMaxScaler(feature_range=(0,1))
 
# Preprocessing the dataset
X_train_scaled = X_scaler.fit_transform(X_train)
X_test_scaled = X_scaler.fit_transform(X_test)
 
## One hot encode Y
onehot_encoder = OneHotEncoder(sparse=False)
Y_train_enc = onehot_encoder.fit_transform(Y_train.reshape(-1,1))
Y_test_enc = onehot_encoder.fit_transform(Y_test.reshape(-1,1))

# Define Model Parameters
learning_rate = 0.01
training_epochs = 1000
 
# define the number of neurons
layer_1_nodes = 150
layer_2_nodes = 150
 
# define the number of inputs
num_inputs = X_train_scaled.shape[1]
num_output = len(np.unique(Y_train, axis = 0)) 
 
# Define the layers
with tf.variable_scope('input'):
    X = tf.placeholder(tf.float32, shape= (None, num_inputs))
 
with tf.variable_scope('layer_1'):
    weights = tf.get_variable('weights1', shape=[num_inputs, layer_1_nodes], initializer = tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable('bias1', shape=[layer_1_nodes], initializer = tf.zeros_initializer())
    layer_1_output =  tf.nn.relu(tf.matmul(X, weights) +  biases) 
 
with tf.variable_scope('layer_2'):
    weights = tf.get_variable('weights2', shape=[layer_1_nodes, layer_2_nodes], initializer = tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable('bias2', shape=[layer_2_nodes], initializer = tf.zeros_initializer())
    layer_2_output =  tf.nn.relu(tf.matmul(layer_1_output, weights) + biases)
 
with tf.variable_scope('output'):
    weights = tf.get_variable('weights3', shape=[layer_2_nodes, num_output], initializer = tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable('bias3', shape=[num_output], initializer = tf.zeros_initializer())
    prediction =  tf.matmul(layer_2_output, weights) + biases
 
with tf.variable_scope('cost'):
    Y = tf.placeholder(tf.float32, shape = (None, num_output))#use 1 instead of num output unless one hot encoding??
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = prediction))
 
with tf.variable_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
 
with tf.variable_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(Y, axis =1), tf.argmax(prediction, axis =1) )
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 
# Logging results
with tf.variable_scope("logging"):
    tf.summary.scalar('current_cost', cost)
    tf.summary.scalar('current_accuacy', accuracy)
    summary = tf.summary.merge_all()