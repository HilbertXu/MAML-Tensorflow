'''
    Date: 14th Feb 2020
    Author: HilbertXu
    Abstract: Code for generating meta-train tasks using miniimagenet and ominiglot dataset
              Meta learning is different from general supervised learning
              The basic training element in training process is TASK(N-way K-shot)
              A batch contains several tasks
              tasks: containing N-way K-shot for meta-train, N-way N-query for meta-test
'''

from __future__ import print_function
import csv
import glob
import os
import sys
import random
import numpy as np
from tqdm import tqdm
from tqdm._tqdm import trange
from PIL import Image
import tensorflow as tf
import cv2

META_TRAIN_DIR = '../dataset/miniImagenet/train'
META_VAL_DIR = '../dataset/miniImagenet/test'


class ImageProc:
    def __init__(self):
        self.path_to_image = '../dataset/miniImagenet/'
        all_images = glob.glob(self.path_to_image + '/images/*')
        # Resize images
        with tqdm(total=len(all_images)) as pbar:
            for i, image_file in enumerate(all_images):
                img = Image.open(image_file)
                img = img.resize((84,84), resample=Image.LANCZOS)
                img.save(image_file)
                if i % 500 == 0 and i > 0:
                    pbar.set_description('{} images processed'.format(i))
                    pbar.update(500)


    def set_dir(self):
        os.chdir(self.path_to_image)
        for datatype in ['train', 'test', 'val']:
            if os.path.exists(datatype) is False:
                print ('create /{} directories'.format(datatype))
                os.system('mkdir {}'.format(datatype))
            else:
                print ('Directories /{} already exist'.format(datatype))
            count = len(open(datatype + '.csv', 'r').readlines())
            with open(datatype + '.csv', 'r') as csvfile:
                print ('Reading {}.csv, {} images in total'.format(datatype, count-1))
                reader = csv.reader(csvfile, delimiter=',')
                last_label = ''
                with tqdm(total=count) as pbar:
                    for i, row in enumerate(reader):
                        if i == 0:  # skip the headers
                            continue
                        image_name = row[0]
                        label = row[1]
                        # Set up a folder for a new class
                        if label != last_label:
                            label_dir = datatype + '/' + label + '/'
                            os.system('mkdir -p {}'.format(label_dir))
                            last_label = label
                        os.system('mv images/' + image_name+ ' ' + label_dir)

                        if i % 400 == 0 and i > 0:
                            pbar.set_description('{} {} images moved'.format(datatype, i))
                            pbar.update(500)
        

class TaskGenerator:
    def __init__(self, config):
        '''
        :param n_way: a train task contains images from different N classes
        :param k_shot: k images used for meta-train
        :param k_query: k images used for meta-test
        :param meta_batchsz: the number of tasks in a batch
        :param total_batch_num: the number of batches
        '''
        # For example:
        # 5-way 1-shot 15-query
        self.meta_batchsz = config['meta_batchsz']
        self.img_num = config['k_shot'] + config['k_query']
        self.n_way = config['n_way']
        self.img_size = config['img_size']
        self.dim_input = np.prod(self.img_size)
        self.dim_output = config['n_way']

        self.metatrain_folders = [os.path.join(META_TRAIN_DIR, label) \
                                    for label in os.listdir(META_TRAIN_DIR) \
                                        if os.path.isdir(os.path.join(META_TRAIN_DIR, label))
                                    ]
        self.metaval_folders = [os.path.join(META_VAL_DIR, label) \
                                    for label in os.listdir(META_VAL_DIR) \
                                        if os.path.isdir(os.path.join(META_VAL_DIR, label))
                                    ]
    def read_images(self, image_file):
        return np.reshape(cv2.imread(image_file).astype(np.float32)/255, self.dim_input)
    
    def convert_to_tensor(self, numpy_objects):
        return (tf.convert_to_tensor(obj) for obj in numpy_objects)
    
    def get_images_and_labels(self, path, label, num_samples=None, shuffle=True):
        if num_samples is not None:
            sampler = lambda x: random.sample(x, num_samples)
        else:
            sampler = lambda x: x 
        images_and_labels = [(i, self.read_images(os.path.join(path, image))) \
                            for i, path in zip(label, path) \
                                for image in sampler(os.listdir(path))
                        ]
        labels = []
        images = []
        for element in images_and_labels:
            labels.append(element[0])
            images.append(element[1])
        return images, labels

    def batch(self, training=True):
        # Generate image tensor of size [4, 80, 84*84*3]
        # Generate label tensor of size [4, 80, 5]
        if training:
            folders = self.metatrain_folders
        else:
            folders = self.metaval_folders
        image_batch=[]
        label_batch=[]

        for i in range(self.meta_batchsz):
            # 16 in one class, 16*5 in one task
            images_per_batch = self.n_way * self.img_num
            # Generate batch
            sampled_folders = random.sample(folders, self.n_way)
            random.shuffle(sampled_folders)
            images, labels = self.get_images_and_labels(sampled_folders, label=range(self.n_way), num_samples=self.img_num)
            image_batch.append(images)
            label_batch.append(labels)
        
        tensor_images, tensor_labels = self.convert_to_tensor((image_batch, label_batch))
        return tensor_images, tf.one_hot(tensor_labels, self.n_way)

def generate_dataset(train_size=200000, test_size=600, config=None):
    def _generate_dataset(size):
        ds = []
        print ('generating dataset of size {}'.format(size))
        with tqdm(total = size) as pbar:
            for i in range(size):
                ds.append(TaskGenerator(config))
                if i % 100 == 0 and i > 0:
                    pbar.set_description('{} dataset generated'.format(i))
                    pbar.update(100)
        return ds  
    return _generate_dataset(train_size), _generate_dataset(test_size)

if __name__ == '__main__':
    # proc = ImageProc()
    # proc.set_dir()
    train_config = {
              'n_way':5, 
              'k_shot':1,
              'k_query':15, 
              'img_size':(84, 84, 3),
              'meta_batchsz':4
             }
    Task = TaskGenerator(train_config)
    images, labels = Task.batch()
    print (labels)
    print (images)
    
    # print (config['n_way'])
                
