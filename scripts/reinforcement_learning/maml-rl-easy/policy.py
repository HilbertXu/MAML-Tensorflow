import tensorflow as tf
import numpy as np 
from collections import OrderedDict

'''
PolicyGradientModel for maze env
'''

def clone_policy(policy, params=None, with_name=False):
    if params is None:
        params = policy.get_trainable_variables
    
    assert isinstance(policy, PolicyGradientModel)

    cloned_policy = PolicyGradientModel(
        input_dim=policy.input_dim,
        output_dim=policy.output_dim,
        hidden_size=policy.hidden_size,
        name=policy.name
    )
    print (cloned_policy.name)

    if with_name:
        cloned_policy.set_params_with_name(params)
    else:
        cloned_policy.set_params(params)
    
    return cloned_policy

class PolicyGradientModel(tf.keras.Model):
    def __init__(self, input_dim=2, output_dim=4, hidden_size=(100,), name=None):
        super(PolicyGradientModel, self).__init__(
            name=name
        )
    
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.nonlinearity = tf.nn.relu
        self.all_param = OrderedDict()
        
        layer_sizes = (self.input_dim,)+self.hidden_size
        self.num_layer = len(self.hidden_size) + 1
        kernel_init = tf.keras.initializers.glorot_uniform()
        bias_init = tf.zeros_initializer()

        for i in range(1, self.num_layer):
            with tf.name_scope('layer_{}'.format(i)):
                kernel = tf.Variable(
                    initial_value=kernel_init(shape=(layer_sizes[i-1], layer_sizes[i]), dtype='float32'),
                    name='kernel',
                    trainable=True
                )
                self.all_param[kernel.name] = kernel
                bias = tf.Variable(
                    initial_value=bias_init(shape=(layer_sizes[i],), dtype='float32'),
                    name='bias',
                    trainable=True
                )
                self.all_param[bias.name] = bias
        
        with tf.name_scope('prob_dist'):
            kernel = tf.Variable(
                initial_value=kernel_init(shape=(layer_sizes[-1], self.output_dim), dtype='float32'),
                name='kernel',
                trainable=True
            )
            self.all_param[kernel.name] = kernel
            bias = tf.Variable(
                    initial_value=bias_init(shape=(self.output_dim,), dtype='float32'),
                    name='bias',
                    trainable=True
                )
            self.all_param[bias.name] = bias

    @property
    def get_trainable_variables(self):
        return list(self.trainable_variables)
    
    def set_params_with_name(self, var_list):
        old_var_list = self.get_trainable_variables
        for (name, var), old_var in zip(var_list.items(), old_var_list):
            old_var.assign(var)

    def set_params(self, var_list):
        old_var_list = self.get_trainable_variables
        for var, old_var in zip(var_list, old_var_list):
            old_var.assign(var)
    
    def update_params(self, grads, step_size=0.01):
        updated_params = OrderedDict()
        params_with_name = [(x.name, x) for x in self.get_trainable_variables]
        for (name, param), grad in zip(params_with_name, grads):
            updated_params[name] = tf.subtract(param, tf.multiply(step_size, grad))
        
        return updated_params
        
    def forward(self, x, params=None):
        if params is None:
            params = self.get_trainable_variables
            params_dict = OrderedDict((v.name, v) for v in params)
        else:
            params_dict = params
        
        x = tf.convert_to_tensor(x)
        for i in range(1, self.num_layer):
            layer_name = self.name + 'layer_{}/'.format(i)
            kernel = params_dict[layer_name+'kernel:0']
            bias = params_dict[layer_name+'bias:0']
            x = tf.matmul(x, kernel)
            x = tf.add(x, bias)
            x = self.nonlinearity(x)
        
        kernel = params_dict[self.name + 'prob_dist/kernel:0']
        bias = params_dict[self.name + 'prob_dist/bias:0']
        x = tf.matmul(x, kernel)
        x = tf.add(x, bias)

        return x
    
    def __call__(self, x, params=None):
        return self.forward(x, params)


if __name__ == '__main__':
    with tf.name_scope('Policy') as scope:
        policy = PolicyGradientModel(name=scope)
    #print(policy.all_param)
    print(type(policy.get_trainable_variables))
    print(policy.name)
    print(policy.get_trainable_variables)
    cloned_policy = clone_policy(policy, policy.all_param, with_name=True)
    print (cloned_policy.get_trainable_variables)




        
    
    