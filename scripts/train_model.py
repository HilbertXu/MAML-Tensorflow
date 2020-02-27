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
import datetime
import numpy as np
import tensorflow as tf 
import time
import matplotlib.pyplot as plt
from task_generator import TaskGenerator
from meta_learner import MetaLearner

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def copy_model(model, x):
    '''
    :param model: model to be copied
    :param x: a set of data, used to build the copied model

    :return copied model
    '''
    copied_model = MetaLearner()
    copied_model(x)
    copied_model.set_weights(model.get_weights())
    return copied_model

def loss_fn(y, pred_y):
    '''
    :param pred_y: Prediction output of model
    :param y: Ground truth

    :return loss value:
    '''
    return tf.reduce_mean(tf.losses.categorical_crossentropy(y, pred_y))

def accuracy_fn(y, pred_y):
    '''
    :param pred_y: Prediction output of model
    :param y: Ground truth

    :return accuracy value:
    '''
    accuracy = tf.keras.metrics.Accuracy()
    _ = accuracy.update_state(tf.argmax(pred_y, axis=1), tf.argmax(y, axis=1))
    return accuracy.result()

def compute_loss(model, x, y, loss_fn=loss_fn):
    '''
    :param model: A neural net
    :param x: Train data
    :param y: Groud truth
    :param loss_fn: Loss function used to compute loss value

    :return Loss value
    '''
    _, pred_y = model(x)
    loss = loss_fn(y, pred_y)
    return loss, pred_y

def compute_gradients(model, x, y, loss_fn=loss_fn):
    '''
    :param model: Neural network
    :param x: Input tensor
    :param y: Ground truth of input tensor
    :param loss_fn: loss function

    :return Gradient tensor
    '''
    with tf.GradientTape() as tape:
        _, pred = model(x)
        loss = loss_fn(y, pred)
        grads = tape.gradient(loss, model.trainable_variables)
    return grads

def apply_gradients(optimizer, gradients, variables):
    '''
    :param optimizer: optimizer, Adam for task-level update, SGD for meta level update
    :param gradients: gradients
    :param variables: trainable variables of model

    :return None
    '''
    optimizer.apply_gradients(zip(gradients, variables))

def regular_train_step(model, x, y, optimizer):
    gradients = compute_gradients(model, x, y, loss_fn=loss_fn)
    apply_gradients(optimizer, gradients, model.trainable_variables)
    return model

def maml_train(model, batch_generator, config=None):
    if config is not None:
        print ('Loading training configuration')
        # problem set up
        n_way = config['n_way']
        k_shot = config['k_shot']
        k_query = config['k_query']
        img_size = config['img_size']
        # training parameters
        total_steps = config['total_steps']
        print_steps = config['print_steps']
        epochs = config['epochs']
        log_steps = config['log_steps']
        inner_lr = config['inner_lr']
        meta_lr = config['meta_lr']
        meta_batchsz = config['meta_batchsz']
        update_steps = config['update_steps']
        print ('Start training process of {}-way {}-shot {}-query problem'.format(n_way, k_shot, k_query))
        print ('{} Epochs, {} steps, inner_lr: {}, meta_lr:{}, meta_batchsz:{}'.format(epochs, total_steps, inner_lr, meta_lr, meta_batchsz))
    else:
        print ('No config input, set to default config')
        # problem set up
        n_way = 5
        k_shot = 1
        k_query = 15
        img_size = [84, 84, 3]
        # training parameters
        total_steps = 10000
        print_steps = 500.0
        epochs = 3
        log_steps = 1000
        inner_lr = 0.01
        meta_lr = 1e-3
        meta_batchsz = 4
        update_steps = 5
        print ('Start training process of {}-way {}-shot {}-query'.format(n_way, k_shot, k_query))
    # Initialize Tensorboard writer
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = '../logs/' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)

    # Initialize Optimizer
    # Meta optimizer for update model parameters
    meta_optimizer = tf.keras.optimizers.Adam(learning_rate=meta_lr, name='meta_optimizer')
    # Manually apply Gradient descent as the task-level optimizer
    # According to MetaLearner.update_fast_weights
    # Initialize Checkpoint handle
    checkpoint = tf.train.Checkpoint(maml_model=model)
    # Set up train record
    query_losses, query_accs = [], []

    # Define the maml train step
    def maml_train_step(batch_set):
        # Set up recorders for every batch
        batch_loss = [0 for _ in range(meta_batchsz)]
        batch_acc = [0 for _ in range(meta_batchsz)]
        # Set up outer gradient tape, only watch model.trainable_variables
        # Because GradientTape only auto record tranable_variables of model
        # But the copied_model.inner_weights is tf.Tensor, so they won't be automatically watched
        with tf.GradientTape() as outer_tape:
            # Set up copied model
            copied_model = model
            # Use the average loss over all tasks in one batch to compute gradients
            for idx, task in enumerate(batch_set):
                # Slice task to support set and query set
                support_x, support_y, query_x, query_y = task
                # Update fast weights several times
                for i in range(update_steps):
                    # Set up inner gradient tape, watch the copied_model.inner_weights
                    with tf.GradientTape(watch_accessed_variables=False) as inner_tape:
                        # we only want inner tape watch the fast weights in each update steps
                        inner_tape.watch(copied_model.inner_weights)
                        inner_loss, _ = compute_loss(copied_model, support_x, support_y)
                    inner_grads = inner_tape.gradient(inner_loss, copied_model.inner_weights)
                    copied_model = MetaLearner.meta_update(copied_model, alpha=inner_lr, grads=inner_grads)
                # Compute task loss & accuracy on the query set
                task_loss, task_pred = compute_loss(copied_model, query_x, query_y, loss_fn=loss_fn)
                task_acc = accuracy_fn(query_y, task_pred)
                batch_loss[idx] += task_loss
                batch_acc[idx] += task_acc
            print (batch_loss, batch_acc)
            # Compute mean loss of the whole batch
            mean_loss = tf.reduce_mean(batch_loss)
        # Compute second order gradients
        outer_grads = outer_tape.gradient(mean_loss, model.inner_weights)
        apply_gradients(meta_optimizer, outer_grads, model.inner_weights)
        # Return reslut of one maml train step
        return batch_loss, batch_acc
            
    # Main loop
    for epoch in range(epochs):
        start = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        print ('[EPOCH. {}] Start at {}'.format(epoch+1, start))
        # Get a batch data
        batch_set = batch_generator.batch()
        # For each epoch update model total_steps times
        for step in range(total_steps):
            # Run maml train step
            print ('[STEP. {}]'.format(step))
            batch_loss, batch_acc = maml_train_step(batch_set)
            # Print train result
            if step % print_steps == 0 and step > 0:
                batch_loss = [loss.numpy() for loss in batch_loss]
                batch_acc = [acc.numpy() for acc in batch_acc]
                print ('[STEP. {}] Task Losses: {}; Task Accuracies: {}'.format(step, batch_loss, batch_acc))
                # Uncomment to see the sampled folders of each task
                # train_ds.print_label_map()
            # Save checkpoint
            if step % log_steps == 0 and step > 0:
                checkpoint.save('../weights/maml_model.ckpt')
    
    return model

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
              'img_size':[84, 84, 3],
              'inner_lr':0.01,
              'meta_lr':1e-3,
              'meta_batchsz':4,
              'update_steps': 5,
              'total_steps':100,
              'log_steps':20,
              'print_steps':5,
              'epochs':3
             }
    test_config = {
              'mode':'test',
              'dataset':'miniimagenet',
              'n_way':5, 
              'k_shot':1,
              'k_query':15, 
              'meta_batchsz':4,
              'img_size':[84, 84, 3],
              'num_steps':(0, 1, 5, 10, 15, 20, 40, 80, 100)
             }
    print ('Initialize model')
    model = MetaLearner()
    model = MetaLearner.initialize(model)

    batch_generator = TaskGenerator(train_config)
    maml_train(model, batch_generator, train_config)
    

    


   