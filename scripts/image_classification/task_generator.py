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

META_TRAIN_DIR = '../../dataset/miniImagenet/train'
META_VAL_DIR = '../../dataset/miniImagenet/test'


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
    def __init__(self, args):
        '''
        :param mode: train or test
        :param n_way: a train task contains images from different N classes
        :param k_shot: k images used for meta-train
        :param k_query: k images used for meta-test
        :param meta_batchsz: the number of tasks in a batch
        :param total_batch_num: the number of batches
        '''
        # For example:
        # 5-way 1-shot 15-query
        if args.dataset == 'miniimagenet':
            self.mode = args.mode
            self.meta_batchsz = args.meta_batchsz
            self.n_way = args.n_way
            self.spt_num = args.k_shot
            self.qry_num = args.k_query
            self.img_size = args.img_size
            self.img_channel = args.img_channel
            self.dim_output = self.n_way
        
        if args.dataset == 'ominiglot':
            self.mode = args.mode
            self.meta_batchsz = args.meta_batchsz
            self.n_way = args.n_way
            self.spt_num = args.k_shot
            self.qry_num = args.k_query
            self.img_size = args.img_size
            self.img_channel = args.img_channel
            self.dim_output = self.n_way
            
        # Set sample folders
        self.metatrain_folders = [os.path.join(META_TRAIN_DIR, label) \
                                    for label in os.listdir(META_TRAIN_DIR) \
                                        if os.path.isdir(os.path.join(META_TRAIN_DIR, label))
                                    ]
        self.metaval_folders = [os.path.join(META_VAL_DIR, label) \
                                    for label in os.listdir(META_VAL_DIR) \
                                        if os.path.isdir(os.path.join(META_VAL_DIR, label))
                                    ]
        # Record the relationship between image label and the folder name in each task
        self.label_map = []
    
    def print_label_map(self):
        print ('[TEST] Label map of current Batch')
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
                    
    def shuffle_set(self, set_x, set_y):
        # Shuffle
        set_seed = random.randint(0, 100)
        random.seed(set_seed)
        random.shuffle(set_x)
        random.seed(set_seed)
        random.shuffle(set_y)
        return set_x, set_y

    def read_images(self, image_file):
        return np.reshape(cv2.imread(image_file).astype(np.float32)/255, (self.img_size, self.img_size, self.img_channel))
    
    def convert_to_tensor(self, np_objects):
        return [tf.convert_to_tensor(obj) for obj in np_objects]
    
    def generate_set(self, folder_list, k_shot=1, k_query=15, shuffle=True):
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
            spt_x = []
            spt_y = []
            qry_x = []
            qry_y = []
            # 此处是从每类的16张图片中抽取第一张作为support set， 其余15张作为query set
            # 并且按照图片路径读取图片，对label进行one hot编码
            # 将support set和query set整体转化为张量
            for i, class_elem in enumerate(ds):
                for j, image_elem in enumerate(class_elem):
                    if j == 0:
                        image = self.read_images(image_elem[0])
                        spt_x.append(image)
                        label = tf.one_hot(image_elem[1], self.n_way)
                        spt_y.append(label)
                    else:
                        image = self.read_images(image_elem[0])
                        qry_x.append(image)
                        label = tf.one_hot(image_elem[1], self.n_way)
                        qry_y.append(label)
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