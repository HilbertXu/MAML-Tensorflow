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
    gradients, loss = compute_gradients(model, x, y)
    apply_gradients(optimizer, gradients, model.trainable_variables)
    return loss

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
    optimizer = tf.keras.optimizers.Adam()

    for epoch in range(epochs):
        print ('[EPOCH. {}]'.format(epoch+1))
        # 200000 steps in total
        # Using 200000 different task generator to generate batch tasks
        # For each batch, containing 4 N-way K-shot tasks
        # For each task, support set & query set
        for idx, batch_ds in enumerate(random.sample(train_ds, len(train_ds))):
            batch_set = batch_ds.batch()

            # update parameters per batch
            for i in range(len(batch_set)):
                support_x, support_y, query_x, query_y = batch_set[i]
                spt_loss = compute_loss(model, support_x, support_y, loss_fn=loss_fn)



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
    # image = cv2.imread('../test/test.jpg').astype(np.float32)/255
    # image = tf.convert_to_tensor(image)
    # image = tf.reshape(image, [-1,84,84,3])
    # label = [1,0,0,0,0]
    # label = tf.convert_to_tensor(label)
    # label = tf.reshape(label, [1, 5])
    # print (tf.argmax(label, axis=1))
    # logits, pred = model.forward(image)
    # print (pred)
    # print (tf.argmax(pred, axis=1))
    train_ds, test_ds = generate_dataset(train_size=1, test_size=1,config=train_config)
    batch_set = train_ds[0].batch()
    support_x, support_y, query_x, query_y = batch_set[0]
    loss, acc = compute_loss(model, support_x, support_y, loss_fn=loss_fn)
    # print (loss)
    # trainable_variables = model._get_trainable_variables()
    # print (model.trainable_variables)
    grads = compute_gradients(model, support_x, support_y, loss_fn=loss_fn)
    # print (model.trainable_variables)
    # print (grads)
    k=0
    model_copy = copy_model(model, support_x)
    # train with bn layers
    for j in range(len(model.layers)):
        if 'conv' in model.layers[j].name or 'dense' in model.layers[j].name:
            model_copy.layers[j].kernel = tf.subtract(model.layers[j].kernel, tf.multiply(inner_lr, grads[k]))
            model_copy.layers[j].bias   = tf.subtract(model.layers[j].bias, tf.multiply(inner_lr, grads[k+1]))
            k+=2
        if 'batch_normalization' in model.layers[j].name:
            print ("Update BN layer")
            model_copy.layers[j].gamma = tf.subtract(model.layers[j].gamma, tf.multiply(inner_lr, grads[k]))
            model_copy.layers[j].beta   = tf.subtract(model.layers[j].beta, tf.multiply(inner_lr, grads[k+1]))
            k+=2

    query_loss, query_acc = compute_loss(model_copy, query_x, query_y)
    query_grads = compute_gradients(model, query_x, query_y)
    # apply_gradients(optimizer, )

    # @TODO
    # Update parameters per batch 
    # according to maml.py
    
    