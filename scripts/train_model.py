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
import argparse
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

def maml_train(model, batch_generator):
    total_steps = args.total_steps
    meta_batchsz = args.meta_batchsz
    update_steps = args.update_steps
    ckpt_steps = args.ckpt_steps
    print_steps = args.print_steps
    inner_lr = args.inner_lr
    meta_lr = args.meta_lr
    ckpt_dir = args.ckpt_dir
    print ('Start training process of {}-way {}-shot {}-query problem'.format(args.n_way, args.k_shot, args.k_query))
    print ('{} steps, inner_lr: {}, meta_lr:{}, meta_batchsz:{}'.format(total_steps, inner_lr, meta_lr, meta_batchsz))

    # Initialize Tensorboard writer
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = args.log_dir + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)

    # Meta optimizer for update model parameters
    meta_optimizer = tf.keras.optimizers.Adam(learning_rate=args.meta_lr, name='meta_optimizer')
    
    # Initialize Checkpoint handle
    checkpoint = tf.train.Checkpoint(maml_model=model)
    
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
            # Compute mean loss of the whole batch
            mean_loss = tf.reduce_mean(batch_loss)
        # Compute second order gradients
        outer_grads = outer_tape.gradient(mean_loss, model.inner_weights)
        apply_gradients(meta_optimizer, outer_grads, model.inner_weights)
        # Return reslut of one maml train step
        return batch_loss, batch_acc
            
    # Main loop
    start = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print ('Start at {}'.format(start))
    # Get a batch data
    batch_set = batch_generator.batch()
    # For each epoch update model total_steps times
    for step in range(total_steps):
        # Run maml train step
        batch_loss, batch_acc = maml_train_step(batch_set)
        # Write to Tensorboard
        with summary_writer.as_default():
            tf.summary.scalar('query loss', tf.reduce_mean(batch_loss), step=step)
            tf.summary.scalar('query accuracy', tf.reduce_mean(batch_acc), step=step)
        # Print train result
        if step % print_steps == 0 and step > 0:
            batch_loss = [loss.numpy() for loss in batch_loss]
            batch_acc = [acc.numpy() for acc in batch_acc]
            print ('[STEP. {}] Task Losses: {}; Task Accuracies: {}'.format(step, batch_loss, batch_acc))
            # Uncomment to see the sampled folders of each task
            # train_ds.print_label_map()
        # Save checkpoint
        if step % ckpt_steps == 0 and step > 0:
            checkpoint.save(ckpt_dir+'maml_model.ckpt')
    
    return model

def eval_model(model, batch_generator):
    # Generate a batch data
    batch_set = batch_generator.batch()
    # Print the label map of each task
    batch_generator.print_label_map()
    # Use a copy of current model
    copied_model = model
    # Initialize optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.inner_lr)

    num_steps = (0, 1, 10, 100)
    task_losses = [0, 0, 0, 0]
    task_accs = [0, 0, 0, 0]
    
    # Record test result
    if 0 in num_steps:
        for idx, task in enumerate(batch_set):
            support_x, support_y, query_x, query_y = task
            loss, pred = compute_loss(model, query_x, query_y)
            acc = accuracy_fn(query_y, pred)
            task_losses[idx] += loss.numpy()
            task_accs[idx] += acc.numpy()
        print ('Before any update steps, test result:')
        print ('Task losses: {}'.format(task_losses))
        print ('Task accuracies: {}'.format(task_accs))

    task_losses = [0, 0, 0, 0]
    task_accs = [0, 0, 0, 0]
    # Test for each task
    for idx, task in enumerate(batch_set):
        print ('========== Task {} =========='.format(idx+1))
        support_x, support_y, query_x, query_y = task
        for step in range(1, np.max(num_steps)+1):
            regular_train_step(model, support_x, support_y, optimizer)
            loss, pred = compute_loss(model, query_x, query_y)
            acc = accuracy_fn(query_y, pred)
            # Record result
            if step in num_steps:
                print ('After {} steps update'.format(step))
                print ('Task losses: {}'.format(loss.numpy()))
                print ('Task accs: {}'.format(acc.numpy()))
                print ('---------------------------------')


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--mode', type=str, help='train or test', default='train')
    # Dataset options
    argparse.add_argument('--dataset', type=str, help='Dataset used to train model', default='miniimagenet')
    # Task options
    argparse.add_argument('--n_way', type=int, help='Number of classes used in classification (e.g. 5-way classification)', default=5)
    argparse.add_argument('--k_shot', type=int, help='Number of images in support set', default=1)
    argparse.add_argument('--k_query', type=int, help='Number of images in query set', default=15)
    # Model options
    argparse.add_argument('--img_size', type=int, help='The size of images input neural net (84 for MiniImagenet, 28 for Ominiglot)', default=84)
    argparse.add_argument('--img_channel', type=int, help='Number of channels of input images (3 for MiniImagenet, 1 for Ominiglot)', default=3)
    argparse.add_argument('--num_filters', type=int, help='Number of filters in the convolution layers (32 for MiniImagenet, 64 for Ominiglot)', default=32)
    # Training options
    argparse.add_argument('--meta_batchsz', type=int, help='Number of tasks in one batch', default=4)
    argparse.add_argument('--update_steps', type=int, help='Number of inner gradient updates for each task', default=1)
    argparse.add_argument('--inner_lr', type=float, help='Learning rate of inner update steps, the step size alpha in the algorithm', default=1e-2) # 0.1 for ominiglot
    argparse.add_argument('--meta_lr', type=float, help='Learning rate of meta update steps, the step size beta in the algorithm', default=1e-3)
    argparse.add_argument('--total_steps', type=int, help='Total update steps for each epoch', default=50)
    # Log options
    argparse.add_argument('--ckpt_steps', type=int, help='Number of steps for recording checkpoints', default=10)
    argparse.add_argument('--print_steps', type=int, help='Number of steps for prints result in the console', default=5)
    argparse.add_argument('--log_dir', type=str, help='Path to the log directory', default='../logs/')
    argparse.add_argument('--ckpt_dir', type=str, help='Path to the checkpoint directory', default='../weights/')
    # Generate args
    args = argparse.parse_args()
    
    print ('Initialize model')
    model = MetaLearner(args=args)
    model = MetaLearner.initialize(model)

    batch_generator = TaskGenerator(args)
    model = maml_train(model, batch_generator)
    eval_model(model, batch_generator)
    

    


   