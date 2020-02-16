"""
    Date: Feb 11st 2020
    Author: Hilbert XU
    Abstract: MetaLeaner model
"""
# @TODO
# change model to keras


import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as keras_backend
import os
import numpy as np 


class MetaLearner(tf.keras.Model):
    """
    Meta Learner
    """
    def __init__(self):
        """
        :param: args from main.py
        """
        super().__init__()
        # for miniimagener dataset set conv2d kernel size=[32, 3, 3]
        # for ominiglot dataset set conv2d kernel size=[64, 3, 3]
        self.filters     = 32
        self.img_channel = 3
        self.img_size    = 84
        self.op_channel  = 5
     
    
    def build(self):
        # Default to use Xavier Initializer
        self.input_layer  = tf.keras.layers.Conv2D(input_shape=(-1, self.img_size, self.img_size, self.img_channel), filters=self.filters,
                                                                kernel_size=(3,3), strides=(1,1),
                                                                padding='SAME', kernel_initializer='glorot_normal')
        self.conv2d_layer = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(3,3),
                                                   strides=(1,1), padding='SAME',
                                                   kernel_initializer='glorot_uniform')
        self.max_pooling  = tf.keras.layers.MaxPool2D(pool_size=(2,2))
        self.batch_norm   = tf.keras.layers.BatchNormalization(axis=-1)
        self.flatten      = tf.keras.layers.Flatten()
        self.dense        = tf.keras.layers.Dense(5, activation="softmax")

                                          
    def forward(self, inputs):
        # set layers
        self.build()
        # input conv block
        x = self.input_layer(inputs)
        x = self.batch_norm(x)
        x = tf.keras.activations.relu(x)
        x = self.max_pooling(x)

        # conv block #2
        x = self.conv2d_layer(x)
        x = self.batch_norm(x)
        x = tf.keras.activations.relu(x)
        x = self.max_pooling(x)

        # conv block #3
        x = self.conv2d_layer(x)
        x = self.batch_norm(x)
        x = tf.keras.activations.relu(x)
        x = self.max_pooling(x)

        # conv block #4
        x = self.conv2d_layer(x)
        x = self.batch_norm(x)
        x = tf.keras.activations.relu(x)
        x = self.max_pooling(x)

        # FC layers
        x = self.flatten(x)
        x = self.dense(x)
        return x


def create_model_via_keras():
    model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=[84,84,3]),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', 
                                   padding='same', kernel_initializer='glorot_normal', name='conv_1'),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.MaxPool2D(pool_size=(2,2)),

            tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', 
                                   padding='same', kernel_initializer='glorot_normal', name='conv_2'),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.MaxPool2D(pool_size=(2,2)),

            tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', 
                                   padding='same', kernel_initializer='glorot_normal', name='conv_3'),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.MaxPool2D(pool_size=(2,2)),

            tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', 
                                   padding='same', kernel_initializer='glorot_normal', name='conv_4'),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.MaxPool2D(pool_size=(2,2)),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(5, activation='softmax')
        ], name='maml')
    print (model.summary())
    return model

