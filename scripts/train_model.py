"""
    Date: Feb 11st 2020
    Author: Hilbert XU
    Abstract: Training process and functions
"""
# -*- coding: UTF-8 -*-
import os
import cv2
import sys
import random
import numpy as np
import tensorflow as tf 
import time
import matplotlib.pyplot as plt
from task_generator import generate_dataset
from meta_learner import MetaLearner

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def copy_model(model, x):
    copied_model = MetaLearner()
    copied_model.forward(x)
    copied_model.set_weights(model.get_weights())
    return copied_model

def np_to_tensor(list_of_numpy_objs):
    return (tf.convert_to_tensor(obj) for obj in list_of_numpy_objs)

def loss_fn(y, pred_y):
    # use softmax_cross_entropy_with_logits as loss function
    return tf.reduce_mean(tf.losses.categorical_crossentropy(y, pred_y))

def compute_loss(model, x, y, loss_fn=loss_fn):
    logits, pred_y = model.forward(x)
    acc = compute_accuracy(pred_y, y)
    loss = loss_fn(y, pred_y)
    return loss, acc

def compute_accuracy(pred_y, y):
    accuracy = tf.keras.metrics.Accuracy()
    _ = accuracy.update_state(tf.argmax(pred_y, axis=1), tf.argmax(y, axis=1))
    return accuracy.result().numpy()

def compute_gradients(model, x, y, loss_fn=loss_fn):
    with tf.GradientTape() as tape:
        loss, _ = compute_loss(model, x, y, loss_fn=loss_fn)
    return tape.gradient(loss, model.trainable_variables)

def apply_gradients(optimizer, gradients, variables):
    optimizer.apply_gradients(zip(gradients, variables))

def regular_train_step(input_list):
    model, x, y, optimizer = input_list
    gradients = compute_gradients(model, x, y)
    apply_gradients(optimizer, gradients, model.trainable_variables)
    return model

def maml_train_step(model, train_ds, epochs=3, inner_lr=0.01, meta_batchsz=4, log_step=1000, config=None):
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
    
    # Initialize optimizer
    optimizer = tf.keras.optimizers.Adam()
    # Set up record list
    query_losses, query_accs = [], []
    # main loop
    for epoch in range(epochs):
        start = time.time()
        print ('[EPOCH. {}] Start at {}'.format(epoch+1, start))
        # 200000 steps in total
        # Using 200000 different task generator to generate task batches
        # For each batch, containing 4 N-way K-shot tasks
        # For each task, containing support set & query set
        # For support set: k_shot images
        # For query set: k_query images
        for i, ds in enumerate(train_ds):
            if i % 10 == 0 and i > 0:
                print ('[STEP. {}] current loss: {}, 10 steps for: {} seconds'.format(i, query_losses[-1], time.time()-start))
                # print (query_losses)
                start = time.time()
            batch_sets = ds.batch()
            # Set the spt_tape to be persistent
            # cause we use support_x to update the fast weights each task
            with tf.GradientTape(persistent=True) as spt_tape:  
                with tf.GradientTape() as query_tape:   
                    batch_loss = []
                    batch_acc = []
                    for i, batch_set in enumerate(batch_sets):
                        support_x, support_y, query_x, query_y = batch_set
                        
                        spt_logits, spt_pred = model.forward(support_x)
                        spt_loss = loss_fn(support_y, spt_pred)
                        spt_acc = compute_accuracy(spt_pred, support_y)
                        spt_grads = spt_tape.gradient(spt_loss, model.trainable_variables)
                        # copy a same model and apply the support gradients 
                        # to update the parameters of the copied model
                        # regrad the parameters of the copied model as the "fast weight" 
                        # the 'first order gradients'
                        k = 0
                        copied_model = copy_model(model, support_x)
                        for j in range(len(model.layers)):
                            if 'conv' in model.layers[j].name or 'dense' in model.layers[j].name:
                                copied_model.layers[j].kernel = tf.subtract(model.layers[j].kernel, tf.multiply(inner_lr, spt_grads[k]))
                                copied_model.layers[j].bias   = tf.subtract(model.layers[j].bias, tf.multiply(inner_lr, spt_grads[k+1]))
                                k+=2
                            if 'batch_normalization' in model.layers[j].name:
                                copied_model.layers[j].gamma = tf.subtract(model.layers[j].gamma, tf.multiply(inner_lr, spt_grads[k]))
                                copied_model.layers[j].beta   = tf.subtract(model.layers[j].beta, tf.multiply(inner_lr, spt_grads[k+1]))
                                k+=2
                        # use the query set to compute loss of model with fast weights
                        qry_loss, qry_acc = compute_loss(copied_model, query_x, query_y, loss_fn=loss_fn)
                        batch_loss.append(qry_loss)
                        batch_acc.append(qry_acc)
                        # Record batch outputs
                        query_losses.append(qry_loss)
                        query_accs.append(qry_acc)
                    # Use the mean loss & accuracy of the whole batch
                    batch_loss = tf.reduce_mean(batch_loss)
                    batch_acc = tf.reduce_mean(batch_acc)
                    
                    # update parameters per batch
                    # the second order gradients
                    gradients = query_tape.gradient(batch_loss, model.trainable_variables)
                    apply_gradients(optimizer, gradients, model.trainable_variables)
    acc = plt.plot(query_accs)
    loss = plt.plot(query_losses)
    plots = [acc, loss]
    legends = ['Accuracy', 'Loss']
    plt.legend(plots, legends)
    plt.show()


if __name__ == '__main__':
    n_way = 5
    k_shot = 1
    meta_batchsz=4
    inner_lr = 0.001
	
    train_config = {
              'mode':'train',
              'dataset':'miniimagenet',
              'n_way':5, 
              'k_shot':1,
              'k_query':15, 
              'img_size':(84, 84, 3),
              'meta_batchsz':4
             }
    model = MetaLearner()
    train_ds, test_ds = generate_dataset(train_size=100, test_size=1,config=train_config)
    maml_train_step(model, train_ds, config=train_config)
    
    