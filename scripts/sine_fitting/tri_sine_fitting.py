# -*- coding: UTF-8 -*-
import pretty_errors
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as keras_backend


# Other dependencies
import random
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Reproduction
np.random.seed(102)
tf.keras.backend.set_floatx('float64')


print('Python version: ', sys.version)
print('TensorFlow version: ', tf.__version__)

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('GPU found at: {}'.format(device_name))

class SinusoidSurfaceGenerator():
    def __init__(self, batchsz=10, x_amplitude=None, x_phase=None, y_ampltude=None, y_phase=None):
        """
        Generate a Function like: z = A1*Sin(x-P1) + A2*Cos(x-P2)
        :param x_amplitude: A1
        :param x_phase: P1
        :param y_amplitude: A2
        :param y_phase: P2
        :param batchsz: number of points in a batch

        """
        self.batchsz = batchsz
        self.x_amplitude = x_amplitude if x_amplitude else np.random.uniform(0.1, 5.0)
        self.x_phase = x_phase if x_phase else np.random.uniform(0, np.pi)
        self.y_ampltude = y_ampltude if y_ampltude else np.random.uniform(0.1, 5.0)
        self.y_phase = y_phase if y_phase else np.random.uniform(0, np.pi)
        self.x_vector = self._sample_x()
        self.y_vector = self._sample_y()
    
    def _sample_x(self):
        return np.random.uniform(-5, 5, self.batchsz)
    
    def _sample_y(self):
        return np.random.uniform(-5, 5, self.batchsz)
    
    def f(self, x, y):
        '''
        Sine Surface Function
        '''
        return self.x_amplitude * np.sin(x - self.x_phase) + self.y_ampltude * np.cos(y - self.y_phase)
    
    def batch(self, x_vector=None, y_vector=None, force_new=False):
        '''
        generate a batch of size batchsz
        z = f(x, y)
        :param (x, y): A point array
        :param force_new: if True, resample point array, else, use self.x & self.y
        :return (x, y) array and z vector
        '''
        points = []
        values = []
        if x_vector is None:
            if force_new:
                x_vector = self._sample_x()
            else:
                x_vector = self.x_vector
        if y_vector is None:
            if force_new:
                y_vector = self._sample_y()
            else:
                y_vector = self.y_vector
        for x in x_vector:
            for y in y_vector:
                point = [x, y]
                points.append(point)
                z = self.f(x, y)
                values.append(z)
        return np.array(points), values
    
    def equally_spaced_samples(self, K=None):
        if K is None:
            K = self.batchsz
        return self.batch(x_vector=np.linspace(-5,5,K), y_vector=np.linspace(-5,5,K))
    
    def plot_figure(self, x=None, y=None, z=None):
        '''
        3D surface
        
        '''
        fig = plt.figure()
        axl = plt.gca(projection='3d')
        print ('using sine function: z = {}*Sin(x - {}) + {}*Cos(y - {})'.format(self.x_amplitude, self.x_phase, self.y_ampltude, self.y_phase))
        if x is None and y is None and z is None:
            x_vector=np.linspace(-5,5,100)
            y_vector=np.linspace(-5,5,100)
            # Test batch generator
            # _, _, x_vector, y_vector = self.batch(x_vector=np.linspace(-5,5,self.batchsz), y_vector=np.linspace(-5,5,self.batchsz))
            x, y = np.meshgrid(x_vector, y_vector)
            z = self.f(x, y)
            axl.plot_surface(x, y, z, cmap='Reds')
            plt.show()
        else:
            x, y = np.meshgrid(x, y )
            axl.plot_surface(x, y, z, cmap='Reds')
            plt.show()

def generate_dataset(batchsz, train_size=20000, test_size=10):
        '''
        Generate dataset of size: train_size and test_size
        A dataset is composed of SinusoidGenerators that are able to provide
        a batch (`K`) elements at a time
        '''
        def _generate_dataset(size):
            return [SinusoidSurfaceGenerator(batchsz=batchsz) for _ in range(size)]
        return _generate_dataset(train_size), _generate_dataset(test_size)

class SineModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.hidden1 = keras.layers.Dense(600, input_shape=(2,))
        self.hidden2 = keras.layers.Dense(800)
        self.hidden3 = keras.layers.Dense(1000)
        self.out = keras.layers.Dense(1)
        
    def forward(self, x):
        x = keras.activations.relu(self.hidden1(x))
        x = keras.activations.relu(self.hidden2(x))
        x = self.out(x)
        return x
        
def copy_model(model, x):
    '''Copy model weights to a new model.
    
    Args:
        model: model to be copied.
        x: An input example. This is used to run
            a forward pass in order to add the weights of the graph
            as variables.
    Returns:
        A copy of the model.
    '''
    copied_model = SineModel()
    
    # If we don't run this step the weights are not "initialized"
    # and the gradients will not be computed.
    copied_model.forward(tf.convert_to_tensor(x))
    
    copied_model.set_weights(model.get_weights())
    return copied_model

def loss_function(pred_y, y):
  return keras_backend.mean(keras.losses.mean_squared_error(y, pred_y))

def np_to_tensor(list_of_numpy_objs):
    return (tf.convert_to_tensor(obj) for obj in list_of_numpy_objs)
    

def compute_loss(model, x, y, loss_fn=loss_function):
    logits = model.forward(x)
    mse = loss_fn(y, logits)
    return mse, logits


def compute_gradients(model, x, y, loss_fn=loss_function):
    with tf.GradientTape() as tape:
        loss, _ = compute_loss(model, x, y, loss_fn)
    return tape.gradient(loss, model.trainable_variables), loss


def apply_gradients(optimizer, gradients, variables):
    optimizer.apply_gradients(zip(gradients, variables))

    
def train_batch(x, y, model, optimizer):
    tensor_x, tensor_y = np_to_tensor((x, y))
    gradients, loss = compute_gradients(model, tensor_x, tensor_y)
    apply_gradients(optimizer, gradients, model.trainable_variables)
    return loss

def train_model(dataset, epochs=1, lr=0.01, log_steps=1000):
    model = SineModel()
    # optimizer = keras.optimizers.Adam(learning_rate=lr)
    optimizer = keras.optimizers.SGD(learning_rate=lr)
    for epoch in range(epochs):
        losses = []
        total_loss = 0
        start = time.time()
        for i, sinusoid_generator in enumerate(dataset):
            x, y = sinusoid_generator.batch()
            loss = train_batch(x, y, model, optimizer)
            total_loss += loss
            curr_loss = total_loss / (i + 1.0)
            losses.append(curr_loss)
            
            if i % log_steps == 0 and i > 0:
                print('Step {}: loss = {}, Time to run {} steps = {:.2f} seconds'.format(
                    i, curr_loss, log_steps, time.time() - start))
                start = time.time()
        plt.plot(losses)
        plt.title('Loss Vs Time steps')
        plt.show()
    return model

def eval_sine_test(model, optimizer, x, y, x_test, y_test, num_steps=(0, 1, 10)):
    '''Evaluate how the model fits to the curve training for `fits` steps.
    
    Args:
        model: Model evaluated.
        optimizer: Optimizer to be for training.
        x: Data used for training.
        y: Targets used for training.
        x_test: Data used for evaluation.
        y_test: Targets used for evaluation.
        num_steps: Number of steps to log.
    '''
    fit_res = []
    
    tensor_x_test, tensor_y_test = np_to_tensor((x_test, y_test))
    
    # If 0 in fits we log the loss before any training
    if 0 in num_steps:
        loss, logits = compute_loss(model, tensor_x_test, tensor_y_test)
        fit_res.append((0, logits, loss))
        
    for step in range(1, np.max(num_steps) + 1):
        train_batch(x, y, model, optimizer)
        loss, logits = compute_loss(model, tensor_x_test, tensor_y_test)
        if step in num_steps:
            fit_res.append(
                (
                    step, 
                    logits,
                    loss
                )
            )
    return fit_res


def eval_sinewave_for_test(model, sinusoid_generator=None, num_steps=(0, 1, 10), lr=0.01, plot=True):
    '''Evaluates how the sinewave addapts at dataset.
    
    The idea is to use the pretrained model as a weight initializer and
    try to fit the model on this new dataset.
    
    Args:
        model: Already trained model.
        sinusoid_generator: A sinusoidGenerator instance.
        num_steps: Number of training steps to be logged.
        lr: Learning rate used for training on the test data.
        plot: If plot is True than it plots how the curves are fitted along
            `num_steps`.
    
    Returns:
        The fit results. A list containing the loss, logits and step. For
        every step at `num_steps`.
    '''
    
    if sinusoid_generator is None:
        sinusoid_generator = SinusoidSurfaceGenerator(batchsz=10)
        
    # generate equally spaced samples for ploting
    x_test, y_test = sinusoid_generator.equally_spaced_samples(100)
    
    # batch used for training
    x, y = sinusoid_generator.batch()
    
    # copy model so we can use the same model multiple times
    copied_model = copy_model(model, x)
    
    # use SGD for this part of training as described in the paper
    optimizer = keras.optimizers.SGD(learning_rate=lr)
    
    # run training and log fit results
    fit_res = eval_sine_test(copied_model, optimizer, x, y, x_test, y_test, num_steps)
    
    # plot
    train, = plt.plot(x, y, '^')
    ground_truth, = plt.plot(x_test, y_test)
    plots = [train, ground_truth]
    legend = ['Training Points', 'True Function']
    for n, res, loss in fit_res:
        cur, = plt.plot(x_test, res[:, 0], '--')
        plots.append(cur)
        legend.append(f'After {n} Steps')
    plt.legend(plots, legend)
    plt.ylim(-5, 5)
    plt.xlim(-6, 6)
    if plot:
        plt.show()
    
    return fit_res

if __name__ == '__main__':
    Generator = SinusoidSurfaceGenerator(batchsz=10)
    Generator.plot_figure()
    # Generator.equally_spaced_samples()
    # # Generator.plot_figure()
    # x, y = Generator.batch()
    # print (x, y)
    # tensor_x, tensor_y = np_to_tensor((x, y))
    # # print (tensor_x, tensor_y)
    # model = SineModel()
    
    # # print (output)
    # mse, logits = compute_loss(model, tensor_x, tensor_y)
    # print (mse)
    # print (logits)
    # Generator.plot_figure()

    neural_model = SineModel()
    train_ds, test_ds = generate_dataset(batchsz=40)
    neural_model = train_model(train_ds)


    x_test, y_test = Generator.equally_spaced_samples(20)
    # print (x_test.T[0])

    tensor_x, tensor_y = np_to_tensor((x_test, y_test))
    output = neural_model.forward(tensor_x)
    print (output)
    x_test = x_test.T
    x = x_test[0]
    y = x_test[1]
    z = output.numpy()
    fig = plt.figure()
    axl = plt.gca(projection='3d') 
    x, y = np.meshgrid(x, y)
    axl.plot_surface(x, y, z, cmap='Blues')
    plt.show()
            



