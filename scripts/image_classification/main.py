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

def write_histogram(model, writer, step):
    '''
    :param model: A model
    :param writer: tf.summary writer
    :param step: Current training step
    '''
    with writer.as_default():
        for idx, layer in enumerate(model.layers):
            if 'conv' in layer.name or 'dense' in layer.name:
                tf.summary.histogram(layer.name+':kernel', layer.kernel, step=step)
                tf.summary.histogram(layer.name+':bias', layer.bias, step=step)
            if 'batch_normalization' in layer.name:
                tf.summary.histogram(layer.name+':gamma', layer.gamma, step=step)
                tf.summary.histogram(layer.name+':beta', layer.beta, step=step)


def write_gradient(grads, writer, step, with_bn=True):
    '''
    :param grads: Gradients on query set
    :param writer: tf.summary writer
    :param step: Current training step
    '''
    if with_bn:
        name = [
            'conv_0:kernel_grad', 'conv_0:bias_grad', 'batch_normalization_1:gamma_grad', 'batch_normalization_1:beta_grad',
            'conv_1:kernel_grad', 'conv_1:bias_grad', 'batch_normalization_2:gamma_grad', 'batch_normalization_2:beta_grad',
            'conv_2:kernel_grad', 'conv_2:bias_grad', 'batch_normalization_3:gamma_grad', 'batch_normalization_3:beta_grad',
            'conv_3:kernel_grad', 'conv_3:bias_grad', 'batch_normalization_4:gamma_grad', 'batch_normalization_4:beta_grad',
            'dense:kernel_grad', 'dense:bias_grad'
        ]
        with writer.as_default():
            for idx, grad in enumerate(grads):
                tf.summary.histogram(name[idx], grad, step=step)
    elif with_bn is False:
        name = [
            'conv_0:kernel_grad', 'conv_0:bias_grad', 'conv_1:kernel_grad', 'conv_1:bias_grad',
            'conv_2:kernel_grad', 'conv_2:bias_grad', 'conv_3:kernel_grad', 'conv_3:bias_grad',
            'dense:kernel_grad', 'dense:bias_grad'
        ]
        with writer.as_default():
            for idx, grad in enumerate(grads):
                tf.summary.histogram(name[idx], grad, step=step)


def restore_model(model, weights_dir):
    '''
    :param model: Model to be restored
    :param weights_dir: Path to weights

    :return: model with trained weights
    '''
    print ('Relod weights from: {}'.format(weights_dir))
    ckpt = tf.train.Checkpoint(maml_model=model)
    latest_weights = tf.train.latest_checkpoint(weights_dir)
    ckpt.restore(latest_weights)
    return model

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
    # Set parameters
    visual = args.visual
    n_way = args.n_way
    k_shot = args.k_shot
    total_batches = args.total_batches
    meta_batchsz = args.meta_batchsz
    update_steps = args.update_steps
    update_steps_test = args.update_steps_test
    test_steps = args.test_steps
    ckpt_steps = args.ckpt_steps
    print_steps = args.print_steps
    inner_lr = args.inner_lr
    meta_lr = args.meta_lr
    ckpt_dir = args.ckpt_dir + args.dataset+'/{}way{}shot/'.format(n_way, k_shot)
    print ('Start training process of {}-way {}-shot {}-query problem'.format(args.n_way, args.k_shot, args.k_query))
    print ('{} steps, inner_lr: {}, meta_lr:{}, meta_batchsz:{}'.format(total_batches, inner_lr, meta_lr, meta_batchsz))

    # Initialize Tensorboard writer
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = args.log_dir + args.dataset +'/{}way{}shot/'.format(n_way, k_shot) + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)

    # Meta optimizer for update model parameters
    meta_optimizer = tf.keras.optimizers.Adam(learning_rate=args.meta_lr, name='meta_optimizer')
    
    # Initialize Checkpoint handle
    checkpoint = tf.train.Checkpoint(maml_model=model)
    losses = []
    accs = []
    test_losses = []
    test_accs = []

    test_min_losses = []
    test_max_accs = []

    def _maml_finetune_step(test_set):
        # Set up recorders for test batch
        batch_loss = [0 for _ in range(meta_batchsz)]
        batch_acc = [0 for _ in range(meta_batchsz)]
        # Set up copied models
        copied_model = MetaLearner.hard_copy(model, args)
        for idx, task in enumerate(test_set):
            # Slice task to support set and query set
            support_x, support_y, query_x, query_y = task
            # Update fast weights several times
            for i in range(update_steps_test):
                # Set up inner gradient tape, watch the copied_model.inner_weights
                with tf.GradientTape(watch_accessed_variables=False) as inner_tape:
                    # we only want inner tape watch the fast weights in each update steps
                    inner_tape.watch(copied_model.inner_weights)
                    inner_loss, _ = compute_loss(copied_model, support_x, support_y)
                inner_grads = inner_tape.gradient(inner_loss, copied_model.inner_weights)
                copied_model = MetaLearner.meta_update(copied_model, args, alpha=inner_lr, grads=inner_grads)
            # Compute task loss & accuracy on the query set
            task_loss, task_pred = compute_loss(copied_model, query_x, query_y, loss_fn=loss_fn)
            task_acc = accuracy_fn(query_y, task_pred)
            batch_loss[idx] += task_loss
            batch_acc[idx] += task_acc

        # Delete copied_model for saving memory
        del copied_model

        return batch_loss, batch_acc
    
    # Define the maml train step
    def _maml_train_step(batch_set):
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
                if visual:
                    with summary_writer.as_default():
                        tf.summary.image('Support Images', support_x, max_outputs=5, step=step)
                        tf.summary.image('Query Images', query_x, max_outputs=5, step=step)
                # Update fast weights several times
                for i in range(update_steps):
                    # Set up inner gradient tape, watch the copied_model.inner_weights
                    with tf.GradientTape(watch_accessed_variables=False) as inner_tape:
                        # we only want inner tape watch the fast weights in each update steps
                        inner_tape.watch(copied_model.inner_weights)
                        inner_loss, _ = compute_loss(copied_model, support_x, support_y)
                    inner_grads = inner_tape.gradient(inner_loss, copied_model.inner_weights)
                    copied_model = MetaLearner.meta_update(copied_model, args, alpha=inner_lr, grads=inner_grads)
                # Compute task loss & accuracy on the query set
                task_loss, task_pred = compute_loss(copied_model, query_x, query_y, loss_fn=loss_fn)
                task_acc = accuracy_fn(query_y, task_pred)
                batch_loss[idx] += task_loss
                batch_acc[idx] += task_acc
            # Compute mean loss of the whole batch
            mean_loss = tf.reduce_mean(batch_loss)
        # Compute second order gradients
        outer_grads = outer_tape.gradient(mean_loss, model.trainable_variables)
        apply_gradients(meta_optimizer, outer_grads, model.trainable_variables)
        if visual:
            # Write gradients histogram
            write_gradient(outer_grads, summary_writer, step)
        # Return reslut of one maml train step
        return batch_loss, batch_acc
            
    # Main loop
    start = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print ('Start at {}'.format(start))
    # For each epoch update model total_batches times
    start = time.time()
    for step in range(total_batches+1):
        # Get a batch data
        batch_set = batch_generator.train_batch()
        # batch_generator.print_label_map()
        # Run maml train step
        batch_loss, batch_acc = _maml_train_step(batch_set)
        if visual:
            # Write histogram
            write_histogram(model, summary_writer, step)
        # Record Loss
        losses.append(tf.reduce_mean(batch_loss).numpy())
        accs.append(tf.reduce_mean(batch_acc).numpy())
        # Write to Tensorboard
        with summary_writer.as_default():
            tf.summary.scalar('query loss', tf.reduce_mean(batch_loss), step=step)
            tf.summary.scalar('query accuracy', tf.reduce_mean(batch_acc), step=step)
        
        # Print train result
        if step % print_steps == 0 and step > 0:
            batch_loss = [loss.numpy() for loss in batch_loss]
            batch_acc = [acc.numpy() for acc in batch_acc]
            print ('[STEP. {}] Task Losses: {}; Task Accuracies: {}; Time to run {} Steps: {}'.format(step, batch_loss, batch_acc, print_steps, time.time()-start))
            start = time.time()
            # Uncomment to see the sampled folders of each task
            # train_ds.print_label_map()

        # Save checkpoint
        if step % ckpt_steps == 0 and step > 0:
            checkpoint.save(ckpt_dir+'maml_model.ckpt')

        # Evaluating model
        if step % test_steps == 0 and step > 0:
            test_set = batch_generator.test_batch()
            batch_generator.print_label_map()
            test_loss, test_acc = _maml_finetune_step(test_set)
            with summary_writer.as_default():
                tf.summary.scalar('test loss', tf.reduce_mean(test_loss), step=step)
                tf.summary.scalar('test accuracy', tf.reduce_mean(test_acc), step=step)
            # Tensor to list            
            test_loss = [loss.numpy() for loss in test_loss]
            test_acc = [acc.numpy() for acc in test_acc]
            # Record test history
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            print ('Test Losses: {}, Test Accuracys: {}'.format(test_loss, test_acc))
            print ('=====================================================================')
        # Meta train step            
    
    # Record training history
    os.chdir(args.his_dir)
    losses_plot, = plt.plot(losses, label = "Train Acccuracy", color='coral')
    accs_plot, = plt.plot(accs,'--',label = "Train Loss", color='royalblue')
    # accs_plot = plt.plot(accs, '--',color='blue')
    plt.legend([losses_plot, accs_plot], ['Train Loss', 'Train Accuracy'])
    plt.title('{} {}-Way {}-Shot MAML Training Process'.format(args.dataset, n_way, k_shot))
    plt.savefig('{}-{}-way-{}-shot.png'.format(args.dataset, n_way, k_shot))

    train_hist = '{}-{}-way{}-shot-train.txt'.format(args.dataset, n_way,k_shot)
    acc_hist = '{}-{}-way{}-shot-acc.txt'.format(args.dataset, n_way,k_shot)
    test_acc_hist = '{}-{}-way{}-shot-acc-test.txt'.format(args.dataset, n_way,k_shot)
    test_loss_hist = '{}-{}-way{}-shot-loss-test.txt'.format(args.dataset, n_way,k_shot)

    # Save History
    f = open(train_hist, 'w')
    for i in range(len(losses)):
        f.write(str(losses[i]) + '\n')
    f.close()

    f = open(acc_hist, 'w')
    for i in range(len(accs)):
        f.write(str(accs[i]) + '\n')
    f.close()

    f = open(test_acc_hist, 'w')
    for i in range(len(test_accs)):
        f.write(str(test_accs[i]) + '\n')
    f.close()

    f = open(test_loss_hist, 'w')
    for i in range(len(test_losses)):
        f.write(str(test_losses[i]) + '\n')
    f.close()

    return model

def eval_model(model, batch_generator, num_steps=None):
    if num_steps is None:
        num_steps = (0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
    # Generate a batch data
    batch_set = batch_generator.test_batch()
    # Print the label map of each task
    batch_generator.print_label_map()
    # Use a copy of current model
    copied_model = model
    # Initialize optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.inner_lr)
    
    task_losses = [0, 0, 0, 0]
    task_accs = [0, 0, 0, 0]

    loss_res = [[] for _ in range(len(batch_set))]
    acc_res = [[] for _ in range(len(batch_set))]
    
    # Record test result
    if 0 in num_steps:
        for idx, task in enumerate(batch_set):
            support_x, support_y, query_x, query_y = task
            loss, pred = compute_loss(model, query_x, query_y)
            acc = accuracy_fn(query_y, pred)
            task_losses[idx] += loss.numpy()
            task_accs[idx] += acc.numpy()
            loss_res[idx].append((0, loss.numpy()))
            acc_res[idx].append((0, acc.numpy()))
        print ('Before any update steps, test result:')
        print ('Task losses: {}'.format(task_losses))
        print ('Task accuracies: {}'.format(task_accs))
    # Test for each task
    for idx, task in enumerate(batch_set):
        print ('========== Task {} =========='.format(idx+1))
        support_x, support_y, query_x, query_y = task
        for step in range(1, np.max(num_steps)+1):
            with tf.GradientTape() as tape:
                #regular_train_step(model, support_x, support_y, optimizer)
                loss, pred = compute_loss(model, support_x, support_y)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            # Test on query set
            qry_loss, qry_pred = compute_loss(model, query_x, query_y)
            qry_acc = accuracy_fn(query_y, qry_pred)
            # Record result
            if step in num_steps:
                loss_res[idx].append((step, qry_loss.numpy()))
                acc_res[idx].append((step, qry_acc.numpy()))
                print ('After {} steps update'.format(step))
                print ('Task losses: {}'.format(qry_loss.numpy()))
                print ('Task accs: {}'.format(qry_acc.numpy()))
                print ('---------------------------------')
    
    for idx in range(len(batch_set)):
        l_x=[]
        l_y=[]
        a_x = []
        a_y=[]
        # plt.subplot(2, 2, idx+1)
        plt.figure()
        for j in range(len(num_steps)):
            l_x.append(loss_res[idx][j][0])
            l_y.append(loss_res[idx][j][1])
            a_x.append(acc_res[idx][j][0])
            a_y.append(acc_res[idx][j][1])
        plt.plot(l_x, l_y, 'x', color='coral')
        plt.plot(a_x, a_y, '*', color='royalblue')
        # plt.annotate('Loss After 1 Fine Tune Step: %.2f'%l_y[1], xy=(l_x[1], l_y[1]), xytext=(l_x[1]-0.2, l_y[1]-0.2))
        # plt.annotate('Accuracy After 1 Fine Tune Step: %.2f'%a_y[1], xy=(a_x[1], a_y[1]), xytext=(a_x[1]-0.2, a_y[1]-0.2))
        plt.plot(l_x, l_y, linestyle='--', color='coral')
        plt.plot(a_x, a_y, linestyle='--', color='royalblue')
        plt.xlabel('Fine Tune Step', fontsize=12)
        plt.fill_between(a_x, [a+0.1 for a in a_y], [a-0.1 for a in a_y], facecolor='royalblue', alpha=0.3)
        legend=['Fine Tune Points','Fine Tune Points','Loss', 'Accuracy']
        plt.legend(legend)
        plt.title('Task {} Fine Tuning Process'.format(idx+1))
        plt.show()



if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--mode', type=str, help='train or test', default='train')
    # Dataset options
    argparse.add_argument('--dataset', type=str, help='Dataset used to train model', default='miniimagenet')
    argparse.add_argument('--visual', type=bool, help='Set True to visualize the batch data', default=True)
    # Task options
    argparse.add_argument('--n_way', type=int, help='Number of classes used in classification (e.g. 5-way classification)', default=5)
    argparse.add_argument('--k_shot', type=int, help='Number of images in support set', default=1)
    argparse.add_argument('--k_query', type=int, help='Number of images in query set(For Omniglot, equal to k_shot)', default=15)
    # Model options
    argparse.add_argument('--num_filters', type=int, help='Number of filters in the convolution layers (32 for MiniImagenet, 64 for Ominiglot)', default=32)
    argparse.add_argument('--with_bn', type=bool, help='Turn True to add BatchNormalization layers in neural net', default=True)
    # Training options
    argparse.add_argument('--meta_batchsz', type=int, help='Number of tasks in one batch', default=4)
    argparse.add_argument('--update_steps', type=int, help='Number of inner gradient updates for each task', default=5)
    argparse.add_argument('--update_steps_test', type=int, help='Number of inner gradient updates for each task while testing', default=10)
    argparse.add_argument('--inner_lr', type=float, help='Learning rate of inner update steps, the step size alpha in the algorithm', default=1e-2) # 0.1 for ominiglot
    argparse.add_argument('--meta_lr', type=float, help='Learning rate of meta update steps, the step size beta in the algorithm', default=1e-3)
    argparse.add_argument('--total_batches', type=int, help='Total update steps for each epoch', default=40000)
    # Log options
    argparse.add_argument('--ckpt_steps', type=int, help='Number of steps for recording checkpoints', default=5000)
    argparse.add_argument('--test_steps', type=int, help='Number of steps for evaluating model', default=5)
    argparse.add_argument('--print_steps', type=int, help='Number of steps for prints result in the console', default=1)
    argparse.add_argument('--log_dir', type=str, help='Path to the log directory', default='../../logs/')
    argparse.add_argument('--ckpt_dir', type=str, help='Path to the checkpoint directory', default='../../weights/')
    argparse.add_argument('--his_dir', type=str, help='Path to the training history directory', default='../../historys/')
    # Generate args
    args = argparse.parse_args()
    
    print ('Initialize model with 4 Conv({} filters) Blocks and 1 Dense Layer'.format(args.num_filters))
    model = MetaLearner(args=args)
    print ('Build model')
    model = MetaLearner.initialize(model)
    model.summary()
    # Initialize task generator
    batch_generator = TaskGenerator(args)

    if args.mode == 'train':
        # model = restore_model(model, '../../weights/{}/{}way{}shot'.format(args.dataset, args.n_way, args.k_shot))
        model = maml_train(model, batch_generator)
    elif args.mode == 'test':
        model = restore_model(model, '../../weights/{}/{}way{}shot'.format(args.dataset, args.n_way, args.k_shot))
        if args.dataset == 'miniimagenet':
            eval_model(model, batch_generator, num_steps=(0, 1, 10, 100, 200, 300, 400, 500, 600))
        elif args.dataset == 'omniglot':
            eval_model(model, batch_generator, num_steps=(0, 1, 5, 100, 200, 300, 400, 500, 600))
