import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp 
from policy import Policy
from collections import OrderedDict

class NormalMLPPolicy(Policy):
    def __init__(
        self, 
        input_size, 
        output_size, 
        hidden_size=(),
        nonlinearity=tf.nn.relu,
        init_std=0.1,
        min_std=1e-6,
        name=None
        ):
        super(NormalMLPPolicy, self).__init__(
            input_size=input_size, 
            output_size=output_size,
            name=name
        )

        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.min_log_ste = tf.math.log(min_std)
        # Hidden layers + output layer
        self.num_layers = len(hidden_size) + 1
        # [input_size, hidden_size]
        layer_sizes = (input_size, ) + hidden_size
        kernel_init = tf.keras.initializers.glorot_uniform()
        bias_init = tf.zeros_initializer()
    
        # Create all parameters
        # 1. Create hidden layers
        for i in range(1, self.num_layers):
            with tf.name_scope('layer_{}'.format(i)):
                kernel = tf.Variable(
                    initial_value=kernel_init(
                        shape=(layer_sizes[i-1], layer_sizes[i]), dtype='float32'
                    ),
                    name='kernel',
                    trainable=True
                )
                self.all_params[kernel.name] = kernel
                bias = tf.Variable(
                    initial_value=bias_init(
                        shape=(layer_sizes[i],), dtype='float32'
                    ),
                    name='bias',
                    trainable=True
                )
                self.all_params[bias.name] = kernel
        
        # 2. Create Probability Distrubition output layer
        with tf.name_scope('pd'):
            kernel = tf.Variable(
                initial_value=kernel_init(
                    shape=(layer_sizes[-1], output_size), dtype='float32'
                ),
                name='kernel',
                trainable=True
            )
            self.all_params[kernel.name] = kernel
            bias = tf.Variable(
                initial_value=bias(
                    shape=(output_size,), dtype='float32'
                ),
                name='bias',
                trainabel=True
            )
            self.all_params[bias.name] = bias
        