import tensorflow as tf 
# import pretty_errors
import random
from tqdm import tqdm
from tqdm._tqdm import trange
import os, sys 
import pickle
import numpy as np 
import cv2
os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from meta_learner import MetaLearner

META_TRAIN_DIR = '../dataset/miniImagenet/train'

# class TaskGenerator:
#     def __init__(self):
#         '''
#         :param mode: train or test
#         :param n_way: a train task contains images from different N classes
#         :param k_shot: k images used for meta-train
#         :param k_query: k images used for meta-test
#         :param meta_batchsz: the number of tasks in a batch
#         :param total_batch_num: the number of batches
#         '''
#         # For example:
#         # 5-way 1-shot 15-query

        
#         self.meta_batchsz = 4
#         self.spt_num = 1
#         self.qry_num = 15
#         self.img_num = 16
#         self.n_way = 5
#         self.img_size = (84, 84, 3)
#         self.dim_input = np.prod(self.img_size)
#         self.dim_output = 5
            
#         # Set sample folders
#         self.metatrain_folders = [os.path.join(META_TRAIN_DIR, label) \
#                                     for label in os.listdir(META_TRAIN_DIR) \
#                                         if os.path.isdir(os.path.join(META_TRAIN_DIR, label))
#                                     ]
#         # Record the relationship between image label and the folder name in each task
#         self.label_map = []
    
#     def sample_folders(self):
#         print ('New generator')
#         random.shuffle(self.metatrain_folders)
#         sampled_folders = random.sample(self.metatrain_folders, self.n_way)
#         for folder in sampled_folders:
#             folder = folder.split('/')[-1]
#             print (folder+'\t', end='')
#         print ('')
        
# def generate_dataset(ds_size=100, config=None):
#     def _generate_dataset(size):
#         ds = []
#         print ('generating dataset of size {}'.format(size))
#         with tqdm(total = size) as pbar:
#             for i in range(size):
#                 ds.append(TaskGenerator())
#                 if i % 100 == 0 and i > 0:
#                     pbar.set_description('{} dataset generated'.format(i))
#                     pbar.update(100)
#         return ds  
#     return _generate_dataset(ds_size)

class TaskGenerator:
    def __init__(self, config):
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
        self.mode = config['mode']
        if self.mode == 'train':
            self.meta_batchsz = config['meta_batchsz']
            self.spt_num = config['k_shot']
            self.qry_num = config['k_query']
            self.img_num = self.spt_num + self.qry_num
            self.n_way = config['n_way']
            self.img_size = config['img_size']
            self.dim_input = np.prod(self.img_size)
            self.dim_output = config['n_way']

        elif self.mode == 'test':
            self.meta_batchsz = config['meta_batchsz']
            self.spt_num = config['k_shot']
            self.qry_num = config['k_query']
            self.img_num = self.spt_num + self.qry_num
            self.n_way = config['n_way']
            self.img_size = config['img_size']
            self.dim_input = np.prod(self.img_size)
            self.dim_output = config['n_way']
            
        # Set sample folders
        self.metatrain_folders = [os.path.join(META_TRAIN_DIR, label) \
                                    for label in os.listdir(META_TRAIN_DIR) \
                                        if os.path.isdir(os.path.join(META_TRAIN_DIR, label))
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
        return np.reshape(cv2.imread(image_file).astype(np.float32)/255, self.img_size)
    
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
        if self.mode == 'train':
            folders = self.metatrain_folders
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
        return batch_set
        

if __name__ == '__main__':
    train_config = {
              'mode':'train',
              'dataset':'miniimagenet',
              'n_way':5, 
              'k_shot':1,
              'k_query':15, 
              'img_size':[84, 84, 3],
              'meta_batchsz':4
             }
    train_ds = TaskGenerator(train_config)
    for i in range(10):
        train_ds.batch()
        train_ds.print_label_map()
    # for i, ds in enumerate(train_ds):
    #     print ('========== This is No.{} task generator ==========='.format(i))
    #     ds.batch()
    #     ds.print_label_map()


	