import tensorflow as tf 
# import pretty_errors
import random
import tqdm
import os, sys 
import pickle
import numpy as np 
import cv2
os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from meta_learner import MetaLearner
from task_generator import generate_dataset
import sinusoid_generator

META_TRAIN_DIR = '../dataset/miniImagenet/train'

def np_to_tensor(list_of_numpy_objs):
    return (tf.convert_to_tensor(obj) for obj in list_of_numpy_objs)

def get_images(paths, labels, nb_samples=None, shuffle=True):
	if nb_samples is not None:
		sampler = lambda x: random.sample(x, nb_samples)
	else:
		sampler = lambda x: x
	images = [(i, os.path.join(path, image)) \
	          for i, path in zip(labels, paths) \
	          for image in sampler(os.listdir(path))]
	if shuffle:
		random.shuffle(images)
	return images

def copy_model(model, x):
	copied_model = MetaLearner()
	copied_model.forward(x)
	copied_model.set_weights(model.get_weights())
	return copied_model

def loss_fn(y, pred_y):
    # use softmax_cross_entropy_with_logits as loss function
    return tf.reduce_mean(tf.losses.categorical_crossentropy(y, pred_y))

def compute_loss(model, x, y, loss_fn=loss_fn):
    logits, pred_y = model.forward(x)
    loss = loss_fn(y, pred_y)
    return logits, pred_y, loss

def compute_accuracy(model, pred_y, y):
    accuracy = tf.keras.metrics.Accuracy(tf.argmax(pred_y, axis=1), tf.argmax(y, axis=1))
    return accuracy

def compute_gradients(model, x, y, loss_fn=loss_fn):
    with tf.GradientTape() as tape:
        logits, pred_y, loss = compute_loss(model, x, y, loss_fn=loss_fn)
    return tape.gradient(loss, model.trainable_variables), loss

def apply_gradients(optimizer, gradients, variables):
    optimizer.apply_gradients(zip(gradients, variables))

def regular_train_step(input_list):
    model, x, y, optimizer = input_list
    gradients, loss = compute_gradients(model, x, y)
    apply_gradients(optimizer, gradients, model.trainable_variables)
    return loss


if __name__ == "__main__":
	n_way = 5
	k_shot = 1
	meta_batchsz=4
	inner_lr = 0.001
	
	train_config = {
              'n_way':5, 
              'k_shot':1,
              'k_query':15, 
              'img_size':(84, 84, 3),
              'meta_batchsz':4
             }
	model = MetaLearner()
	print ("Model Layers: {}")
	# print (model.layers)
	
	task_train, task_test = generate_dataset(train_size=1, test_size=1, config=train_config)
	images, labels = task_train[0].batch()
	print (images.shape)
	images = [tf.reshape(tensor, [-1, 84, 84, 3]) for tensor in tf.unstack(images, axis=0)]
	labels = [tf.reshape(tensor, [-1, 5]) for tensor in tf.unstack(labels, axis=0)]
	support_x = tf.slice(images[0], [0,0,0,0], [n_way * k_shot, -1,-1,-1], name='support_x')
	query_x   = tf.slice(images[0], [n_way * k_shot, 0,0,0], [-1, -1, -1,-1], name='query_x')
	support_y = tf.slice(labels[0], [0,0], [n_way * k_shot,-1], name='support_y')
	query_y   = tf.slice(labels[0], [n_way * k_shot, 0], [-1, -1], name='query_y')
	logtis, pred = model.forward(support_x)
	print ('pred: {}'.format(pred))
	print ('label: {}'.format(support_y))
	loss = loss_fn(pred, support_y)
	with tf.GradientTape() as tape:
		gradients = tape.gradient(loss, model.trainable_variables)
	print ("Model trainable variables: {}".format(len(model._trainable_variables())))
	# print (model._trainable_variables())
	# with open('model_trainable_varibales.txt', 'a') as f:
	print ('Loss: {}'.format(loss))
	print ('Gradients: {}'.format(gradients))
	# print (model.layers[0].kernel)
	# print (len(images))
	# print (images[0].shape)
	# print (labels[0].shape)
	# for i in range(len(model.layers)):
	# 	print (model.layers[i].name)
	
	# batch_spt_loss= []
	# batch_qry_loss= []
	# batch_qry_acc=[]
	# batch_qry_pred=[]
	# # Slice tensors into support tensors and query tensors
	# # support set: support_x [4, 5*1, 84*84*3]
	# #              support_y [4, 5*1, 5]
	# support_x = tf.slice(images[0], [0,0,0,0], [n_way * k_shot, -1,-1,-1], name='support_x')
	# query_x   = tf.slice(images[0], [n_way * k_shot, 0,0,0], [-1, -1, -1,-1], name='query_x')
	# # support set: support_x [4, 5*15, 84*84*3]
	# #              support_y [4, 5*15, 5]
	# support_y = tf.slice(labels[0], [0,0], [n_way * k_shot,-1], name='support_y')
	# query_y   = tf.slice(labels[0], [n_way * k_shot, 0], [-1, -1], name='query_y')
	# with tf.GradientTape() as support_tape:
	# 	with tf.GradientTape() as query_tape:
	# 		_, support_pred, support_loss = compute_loss(model, support_x, support_y)
	# 		# support_acc = compute_accuracy(model, support_pred, support_y)
	# 		batch_spt_loss.append(support_loss)
	# 	support_grads = support_tape.gradient(support_loss, model.trainable_variables)
	# 	k=0
	# 	model_copy = copy_model(model, support_x)
	# 	for j in range(len(model.layers)):
	# 		if 'conv' in model.layers[j].name or 'dense' in model.layers[j].name:
	# 			model_copy.layers[j].kernel = tf.subtract(model.layers[j].kernel, tf.multiply(inner_lr, support_grads[k]))
	# 			model_copy.layers[j].bias   = tf.subtract(model.layers[j].bias, tf.multiply(inner_lr, support_grads[k+1]))
	# 			k+=2
	# 	_, query_pred, query_loss = compute_loss(model_copy, query_x, query_y)
	# 	# query_acc = compute_accuracy(model, query_x, query_y)
	# 	batch_qry_loss.append(query_loss)
	# 	# batch_qry_acc.append(query_acc)
	# 	batch_qry_pred.append(query_pred)
	# for i in range(4):
	# 	# Slice tensors into support tensors and query tensors
	# 	# support set: support_x [4, 5*1, 84*84*3]
	# 	#              support_y [4, 5*1, 5]
	# 	support_x = tf.slice(images[i], [0,0,0,0], [n_way * k_shot, -1,-1,-1], name='support_x')
	# 	query_x   = tf.slice(images[i], [n_way * k_shot, 0,0,0], [-1, -1, -1,-1], name='query_x')
	# 	# support set: support_x [4, 5*15, 84*84*3]
	# 	#              support_y [4, 5*15, 5]
	# 	support_y = tf.slice(labels[i], [0,0], [n_way * k_shot,-1], name='support_y')
	# 	query_y   = tf.slice(labels[i], [n_way * k_shot, 0], [-1, -1], name='query_y')

	# 	print ('Shape of support_x:{} query_x:{}, support_y:{} query_y:{}'.format(support_x.shape, query_x.shape, support_y.shape, query_y.shape))
	# 	with tf.GradientTape() as support_tape:
	# 		with tf.GradientTape() as query_tape:
	# 			_, support_pred, support_loss = compute_loss(model, support_x, support_y)
	# 			# support_acc = compute_accuracy(model, support_x, support_y)
	# 			batch_spt_loss.append(support_loss)
	# 		support_grads = support_tape.gradient(support_loss, model.trainable_variables)
	# 		k=0
	# 		model_copy = copy_model(model, support_x)
	# 		for j in range(len(model.layers)):
	# 			if 'conv' in model.layers[j].name or 'dense' in model.layers[j].name:
	# 				model_copy.layers[j].kernel = tf.subtract(model.layers[j].kernel, tf.multiply(inner_lr, support_grads[k]))
	# 				model_copy.layers[j].bias   = tf.subtract(model.layers[j].bias, tf.multiply(inner_lr, support_grads[k+1]))
	# 				k+=2
	# 		_, query_pred, query_loss = compute_loss(model_copy, query_x, query_y)
	# 		# query_acc = compute_accuracy(model, query_x, query_y)
	# 		batch_qry_loss.append(query_loss)
	# 		# batch_qry_acc.append(query_acc)
	# 		batch_qry_pred.append(query_pred)
	
	

	
	

	