"""
    Date: Feb 11st 2020
    Author: Hilbert XU
    Abstract: MetaLeaner model
"""

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as keras_backend
import os
import numpy as np 
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class MetaLearner(tf.keras.Model):
    """
    Meta Learner
    """
    def __init__(self, filters=32, img_size=[84,84,3], n_way=5, model_name='maml', training=True):
        """
        :param filters: Number of filters in conv layers
        :param img_size: Size of input image, [84, 84, 3] for miniimagenet
        :param n_way: Number of classes
        :param name: Name of model
        """
        super().__init__()
        # for miniimagener dataset set conv2d kernel size=[32, 3, 3]
        # for ominiglot dataset set conv2d kernel size=[64, 3, 3]
        self.filters     = filters
        self.img_size    = img_size
        self.op_channel  = n_way
        self.model_name  = model_name   
        self.training    = training 

        # Build model layers
        self.conv_1 = tf.keras.layers.Conv2D(input_shape=(-1, 84, 84, 3), filters=32, kernel_size=(3,3), strides=(1,1), padding='SAME', kernel_initializer='glorot_normal')
        self.bn_1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.max_pool_1 = tf.keras.layers.MaxPool2D(pool_size=(2,2))

        self.conv_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='SAME', kernel_initializer='glorot_normal')
        self.bn_2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.max_pool_2 = tf.keras.layers.MaxPool2D(pool_size=(2,2))

        self.conv_3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='SAME', kernel_initializer='glorot_normal')
        self.bn_3 = tf.keras.layers.BatchNormalization(axis=-1)
        self.max_pool_3 = tf.keras.layers.MaxPool2D(pool_size=(2,2))

        self.conv_4 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='SAME', kernel_initializer='glorot_normal')
        self.bn_4 = tf.keras.layers.BatchNormalization(axis=-1)
        self.max_pool_4 = tf.keras.layers.MaxPool2D(pool_size=(2,2))

        self.fc = tf.keras.layers.Flatten()
        self.out = tf.keras.layers.Dense(self.op_channel)
        
    
    def forward(self, x):
        # Conv block #1
        x = tf.keras.activations.relu(self.max_pool_1(self.bn_1(self.conv_1(x), training=self.training)))
        # Conv block #2
        x = tf.keras.activations.relu(self.max_pool_2(self.bn_2(self.conv_2(x), training=self.training)))
        # Conv block #3
        x = tf.keras.activations.relu(self.max_pool_3(self.bn_3(self.conv_3(x), training=self.training)))
        # Conv block #4
        x = tf.keras.activations.relu(self.max_pool_4(self.bn_4(self.conv_4(x), training=self.training)))

        # Fully Connect Layer
        x = self.fc(x)
        # Logits Output
        logits = self.out(x)
        # Prediction
        pred = tf.keras.activations.softmax(logits)
        
        return logits, pred
            