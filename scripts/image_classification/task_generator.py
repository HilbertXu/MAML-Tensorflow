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
import argparse
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


        
class TaskGenerator:
    def __init__(self, args=None):
        '''
        :param mode: train or test
        :param n_way: a train task contains images from different N classes
        :param k_shot: k images used for meta-train
        :param k_query: k images used for meta-test
        :param meta_batchsz: the number of tasks in a batch
        :param total_batch_num: the number of batches
        '''
        if args is not None:
            self.dataset = args.dataset
            self.mode = args.mode
            self.meta_batchsz = args.meta_batchsz
            self.n_way = args.n_way
            self.spt_num = args.k_shot
            self.qry_num = args.k_query
            self.dim_output = self.n_way
        else:
            self.dataset = 'miniimagenet'
            self.mode = 'train'
            self.meta_batchsz = 4
            self.n_way = 5
            self.spt_num = 1
            self.qry_num = 15
            self.img_size = 84
            self.img_channel = 3
            self.dim_output = self.n_way
        # For example:
        # 5-way 1-shot 15-query for MiniImagenet
        if self.dataset == 'miniimagenet':
            self.img_size = 84
            self.img_channel = 3
            META_TRAIN_DIR = '../../dataset/miniImagenet/train'
            META_VAL_DIR = '../../dataset/miniImagenet/test'
            # Set sample folders
            self.metatrain_folders = [os.path.join(META_TRAIN_DIR, label) \
                                        for label in os.listdir(META_TRAIN_DIR) \
                                            if os.path.isdir(os.path.join(META_TRAIN_DIR, label))
                                        ]
            self.metaval_folders = [os.path.join(META_VAL_DIR, label) \
                                        for label in os.listdir(META_VAL_DIR) \
                                            if os.path.isdir(os.path.join(META_VAL_DIR, label))
                                        ]
        
        if self.dataset == 'omniglot':
            self.img_size = 28
            self.img_channel = 1
            if self.spt_num != self.qry_num:
                # For Omniglot dataset set k_query = k_shot
                self.qry_num = self.spt_num
            DATA_FOLDER = '../../dataset/omniglot'
            character_folders = [
                os.path.join(DATA_FOLDER, family, character) \
                    for family in os.listdir(DATA_FOLDER) \
                        if os.path.isdir(os.path.join(DATA_FOLDER, family)) \
                            for character in os.listdir(os.path.join(DATA_FOLDER, family))
            ]
            # Shuffle dataset
            random.seed(9314)
            random.shuffle(character_folders)
            # Slice dataset to train set and test set
            # Use 1400 Alphabets as train set, the rest as test set
            self.metatrain_folders = character_folders[:1400]
            self.metaval_folders = character_folders[1400:]   
        
        # Record the relationship between image label and the folder name in each task
        self.label_map = []
    
    def print_label_map(self):
        print ('[TEST] Label map of current Batch')
        if self.dataset == 'miniimagenet':
            if len(self.label_map) > 0:
                for i, task in enumerate(self.label_map):
                    print ('========= Task {} =========='.format(i+1))
                    for i, ref in enumerate(task):
                        path = ref[0]
                        label = path.split('/')[-1]
                        print ('map {} --> {}\t'.format(label, ref[1]), end='')
                        if i == 4:
                            print ('')
                print ('========== END ==========')
                self.label_map = []
            elif len(self.label_map) == 0:
                print ('ERROR! print_label_map() function must be called after generating a batch dataset')
        elif self.dataset == 'omniglot':
            if len(self.label_map) > 0:
                for i, task in enumerate(self.label_map):
                    print ('========= Task {} =========='.format(i+1))
                    for i, ref in enumerate(task):
                        path = ref[0]
                        label = path.split('/')[-2] +'/'+ path.split('/')[-1]
                        print ('map {} --> {}\t'.format(label, ref[1]), end='')
                        if i == 4:
                            print ('')
                print ('========== END ==========')
                self.label_map = []
            elif len(self.label_map) == 0:
                print ('ERROR! print_label_map() function must be called after generating a batch dataset')

                    
    def shuffle_set(self, set_x, set_y):
        # Shuffle
        set_seed = random.randint(0, 100)
        random.seed(set_seed)
        random.shuffle(set_x)
        random.seed(set_seed)
        random.shuffle(set_y)
        return set_x, set_y

    def read_images(self, image_file):
        if self.dataset == 'omniglot':
            # For Omniglot dataset image size:[28, 28, 1]
            return np.reshape(cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2GRAY).astype(np.float32)/255, (self.img_size, self.img_size, self.img_channel))
        if self.dataset == 'miniimagenet':
            # For Omniglot dataset image size:[84, 84, 3]
            return np.reshape(cv2.imread(image_file).astype(np.float32)/255, (self.img_size, self.img_size, self.img_channel))
    
    def convert_to_tensor(self, np_objects):
        return [tf.convert_to_tensor(obj) for obj in np_objects]
    
    def generate_set(self, folder_list, shuffle=True):
        k_shot = self.spt_num
        k_query = self.qry_num
        set_sampler = lambda x: random.sample(x, k_shot+k_query)
        label_map = []
        images_with_labels = []
        # sample images for support set and query set
        # images_with_labels: size [5, 16] 5 classes with 16 images & labels per class
        for i, elem in enumerate(folder_list):
            folder = elem[0]
            label = elem[1]
            label_map.append((folder, label))
            image_with_label = [(os.path.join(folder, image), label) \
                                for image in set_sampler(os.listdir(folder))]
            images_with_labels.append(image_with_label)
        self.label_map.append(label_map)
        if shuffle == True:
            for i, elem in enumerate(images_with_labels):
                random.shuffle(elem)
        
        # Function for slicing the dataset
        # support set & query set
        def _slice_set(ds):
            spt_x = list()
            spt_y = list()
            qry_x = list()
            qry_y = list()
            # 此处是从每类的k_shot+k_query张图片中抽取k_shot张作为support set， 其余作为query set
            # 并且按照图片路径读取图片，对label进行one hot编码
            # 将support set和query set整体转化为张量
            for i, class_elem in enumerate(ds):
                spt_elem = random.sample(class_elem, self.spt_num)
                qry_elem = [elem for elem in class_elem if elem not in spt_elem]
                spt_elem = list(zip(*spt_elem))
                qry_elem = list(zip(*qry_elem))
                spt_x.extend([self.read_images(img) for img in spt_elem[0]])
                spt_y.extend([tf.one_hot(label, self.n_way) for label in spt_elem[1]])
                qry_x.extend([self.read_images(img) for img in qry_elem[0]])
                qry_y.extend([tf.one_hot(label, self.n_way) for label in qry_elem[1]])

            # Shuffle datasets
            spt_x, spt_y = self.shuffle_set(spt_x, spt_y)
            qry_x, qry_y = self.shuffle_set(qry_x, qry_y)
            # convert to tensor
            spt_x, spt_y = self.convert_to_tensor((np.array(spt_x), np.array(spt_y)))
            qry_x, qry_y = self.convert_to_tensor((np.array(qry_x), np.array(qry_y)))
            return spt_x, spt_y, qry_x, qry_y
        return _slice_set(images_with_labels)
              
    def batch(self):
        '''
        :return: a batch of support set tensor and query set tensor
        
        '''
        folder = []
        if self.mode == 'train':
            folders = self.metatrain_folders
        if self.mode == 'test':
            folders = self.metaval_folders
        # Shuffle root folder in order to prevent repeat
        random.shuffle(folder)
        batch_set = []
        # Generate batch dataset
        # batch_spt_set: [meta_batchsz, n_way * k_shot, image_size] & [meta_batchsz, n_way * k_shot, n_way]
        # batch_qry_set: [meta_batchsz, n_way * k_query, image_size] & [meta_batchsz, n_way * k_query, n_way]
        for i in range(self.meta_batchsz):
            random.shuffle(folders)
            sampled_folders = random.sample(folders, self.n_way)
            random.shuffle(sampled_folders)
            folder_with_label = []
            for i, folder in enumerate(sampled_folders):
                elem = (folder, i)
                folder_with_label.append(elem)
            support_x, support_y, query_x, query_y = self.generate_set(folder_with_label)
            batch_set.append((support_x, support_y, query_x, query_y))
        # return [meta_batchsz * (support_x, support_y, query_x, query_y)]
        return batch_set

