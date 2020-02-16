import tensorflow as tf 
# import pretty_errors
import random
import tqdm
import os, sys 
import pickle
import numpy as np 
import cv2
os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'

from meta_learner import MetaLearner, create_model_via_keras
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
    copied_model = create_model_via_keras()
    copied_model(x)
    copied_model.set_weights(model.get_weights())
    return copied_model



if __name__ == "__main__":
	train_config = {
              'n_way':5, 
              'k_shot':1,
              'k_query':15, 
              'img_size':(84, 84, 3),
              'meta_batchsz':4
             }
	# model = MetaLearner()
	model = create_model_via_keras()
	# task_train, task_test = generate_dataset(train_size=1, test_size=1, config=train_config)
	# images, labels = task_train[0].batch()
	image = cv2.imread('test.jpg').astype(np.float32)/255
	image = tf.convert_to_tensor(image)
	image = tf.reshape(image, [-1,84,84,3])
	label = [1,0,0,0,0]
	print (image)
	# pred = model.forward(image)
	pred = model(image)
	# weights = model.get_weights() 
	print (pred)
	copied_model = copy_model(model, image)

	

	# images = tf.stack(images)
	# labels = tf.stack(labels)
	# print (images)
	# print (labels)
	# # support_x : [4, 1*5, 84*84*3]
	# # query_x   : [4, 15*5, 84*84*3]
	# # support_y : [4, 5, 5]
	# # query_y   : [4, 15*5, 5]
	# support_x = tf.slice(images, [0,0,0], [-1, train_config['n_way'] * train_config['k_shot'], -1], name='support_x')
	# query_x = tf.slice(images, [0, train_config['n_way'] * train_config['k_shot'], 0], [-1, -1, -1], name='query_x')
	# support_y = tf.slice(labels, [0,0,0], [-1, train_config['n_way'] * train_config['k_shot'], -1], name='support_x')
	# query_y = tf.slice(labels, [0, train_config['n_way'] * train_config['k_shot'], 0], [-1, -1, -1], name='query_x')

	# print (support_x)
	# print (support_y)
	# print (query_x)
	# print (query_y)
	
	

	