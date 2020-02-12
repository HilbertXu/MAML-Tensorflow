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


class MetaLearner(tf.keras.Model):
    """
    Meta Learner
    """
    def __init__(self, args=None):
        """
        :param: args from main.py
        """
        super().__init__()
        if args.dataset == 'miniimagenet':
            # for miniimagener dataset set conv2d kernel size=[32, 3, 3]
            # for ominiglot dataset set conv2d kernel size=[64, 3, 3]
            self.filters     = 32
            self.img_channel = 3
            self.img_size    = 84
            self.op_channel  = args.n_way
        elif args.dataset == 'ominiglot':
            self.filters     = 64
            self.img_channel = 1
            self.img_size    = 28
            self.op_channel  = args.n_way
     
    
    def build(self):
        # Default to use Xavier Initializer
        self.input_layer  = tf.keras.layers.Conv2D(input_shape=(self.img_size,self.img_size,self.img_channel), filters=self.filters,
                                                                kernel_size=(3,3), strides=(1,1),
                                                                padding='SAME', kernel_initializer='glorot_normal')
        self.conv2d_layer = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(3,3),
                                                   strides=(1,1), padding='SAME',
                                                   kernel_initializer='glorot_uniform')
        self.max_pooling  = tf.keras.layers.MaxPool2D(pool_size=(2,2))
        self.relu         = tf.keras.activations.relu()
        self.batch_norm   = tf.keras.layers.BatchNormalization(axis=-1)
        self.flatten      = tf.keras.layers.Flatten()
        self.dense        = tf.keras.layers.Dense(self.op_channel, activation="softmax")
                                                            
    def forward(self, inputs):
        self.build()
        # input conv block
        x = self.input_layer(inputs)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.max_pooling(x)

        # conv block #2
        x = self.conv2d_layer(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.max_pooling(x)

        # conv block #3
        x = self.conv2d_layer(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.max_pooling(x)

        # conv block #4
        x = self.conv2d_layer(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.max_pooling(x)

        # FC layers
        x = self.dense(x)
        return x


