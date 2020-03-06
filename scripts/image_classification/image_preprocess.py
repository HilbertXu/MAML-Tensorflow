'''
    Date: 1st Feb 2020
    Author: HilbertXu
    Abstract: Code for preprocessing dataset images
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

class ImageProc:
    def __init__(self, dataset):
        if dataset == 'miniimagenet':
            print ('Processing MiniImagenet dataset')
            self.path_to_image = '../../dataset/miniImagenet/'
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
            self.set_dir()
        elif dataset == 'omniglot':
            print ('Processing Omniglot dataset')
            self.root = '../../dataset/omniglot'
            character_folders = [
                os.path.join(self.root, family, character) \
                    for family in os.listdir(self.root) \
                        if os.path.isdir(os.path.join(self.root, family)) \
                            for character in os.listdir(os.path.join(self.root, family))
            ]
            for character in character_folders:
                print ('Currently processing {}'.format(character))
                images = os.listdir(character)
                for image in images:
                    image_file = os.path.join(character, image)
                    img = Image.open(image_file)
                    img = img.resize((28,28), resample=Image.LANCZOS)
                    img.save(image_file)

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

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    # Dataset options
    argparse.add_argument('--dataset', type=str, help='Dataset to be processed', default='miniimagenet')
    # Generate args
    args = argparse.parse_args()
    proc = ImageProc(args.dataset)
