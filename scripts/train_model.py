# -*- coding: UTF-8 -*-
import os
import sys
import random
import numpy as np
import tensorflow as tf 
import time
import matplotlib.pyplot as plt
from task_generator import generate_dataset
from meta_learner import MetaLearner, create_model_via_keras

def copy_model(model, x):
    copied_model = create_model_via_keras()
    copied_model(x)
    copied_model.set_weights(model.get_weights())
    return copied_model

def np_to_tensor(list_of_numpy_objs):
    return (tf.convert_to_tensor(obj) for obj in list_of_numpy_objs)

def loss_fn(y, pred_y):
    # use softmax_cross_entropy_with_logits as loss function
    return tf.reduce_mean(tf.losses.categorical_crossentropy(y, pred_y))

def compute_loss(model, x, y, loss_fn=loss_fn):
    logits, pred_y = model.forward(x)
    loss = loss_fn(y, pred_y)
    return logits, pred_y, loss

def compute_accuracy(model, x, y):
    _, pred_y = model.forward(x)
    accuracy = tf.keras.metrics.Accuracy(tf.argmax(pred_y, axis=1), tf.argmax(y, axis=1))
    return accuracy

def compute_gradients(model, x, y, loss_fn=loss_fn):
    with tf.GradientTape() as tape:
        logits, pred_y, loss = compute_loss(model, x, y, loss_fn=loss_fn)
    return tape.gradient(loss, model.trainable_variables), loss

def apply_gradients(optimizer, gradients, variables):
    optimizer.apply_gradients(zip(gradients, variables))

def train_step(input_list):
    model, x, y, optimizer = input_list
    gradients, loss = compute_gradients(model, x, y)
    apply_gradients(optimizer, gradients, model.trainable_variables)
    return loss

def maml_train(model, train_ds, epochs=6, inner_lr=0.01, batch_size=4, log_steps=1000, config=None):
    # Set training config
    if config is not None:
        print ('load train config')
        n_way = config['n_way']
        k_shot = config['k_shot']
        k_query = config['k_query']
        print ('Start training process of {}-way {}-shot {}-query'.format(n_way, k_shot, k_query))
    else:
        print ('No config input, set to default config')
        n_way = 5
        k_shot = 1
        k_query = 15
        print ('Start training process of {}-way {}-shot {}-query'.format(n_way, k_shot, k_query))
    # Start train process 
    optimizer = tf.keras.optimizers.Adam()
    losses = []
    support_accuracy = []
    query_accuracy = []
    for epoch in epochs:
        total_loss = 0
        start = time.time()
        for i, batch_ds in enumerate(random.sample(train_ds, len(train_ds))):
            # Generate images tensor x of shape (4, 80, 84*84*3)
            # Generate labels tensor y of shape (4, 80, 5)
            x, y =  batch_ds.batch()
            # Slice tensors into support tensors and query tensors
            support_x = tf.slice(x, [0,0,0], [-1, n_way * k_shot, -1], name='support_x')
            query_x   = tf.slice(x, [0, n_way * k_shot, 0], [-1, -1, -1], name='query_x')
            support_y = tf.slice(y, [0,0,0], [-1, n_way * k_shot, -1], name='support_y')
            query_y   = tf.slice(y, [0, n_way * k_shot, 0], [-1, -1, -1], name='query_y')

            model.forward(support_x)
            with tf.GradientTape() as support_tape:
                with tf.GradientTape() as query_tape:
                    support_loss, _ = compute_loss(model, support_x, support_y)
                    support_acc = compute_accuracy(model, support_x, support_y)
                    support_accuracy.append(support_acc)
                support_grads = support_tape.gradient(support_loss, model.trainable_variables)
                k=0
                model_copy = copy_model(model, support_x)
                for j in range(len(model.layers)):
                    model_copy.layers[j].kernel = tf.subtract(model.layers[j].kernel, tf.multiply(inner_lr, support_grads[k]))
                    model_copy.layers[j].bias   = tf.subtract(model.layers[j].bias, tf.multiply(inner_lr, support_grads[k+1]))
                    k+=2
                query_loss, logits = compute_loss(model_copy, query_x, query_y)
                query_acc = compute_accuracy(model_copy, query_x, query_y)
                query_accuracy.append(query_acc)
            query_grads = query_tape.gradient(query_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(query_grads, model.trainable_variables))

            # Record logs
            total_loss += query_loss
            loss = total_loss / (i+1.0)
            losses.append(loss)

            if i % log_steps == 0 and i > 0:
                print('Step {}: loss = {}, Time to run {} steps = {}'.format(i, loss, log_steps, time.time() - start))
                start = time.time()
            
    result = [losses, support_accuracy, query_accuracy]
    return model, result