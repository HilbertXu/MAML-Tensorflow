'''
    Date: 6th Mar 2020
    Author: HilbertXu
    Abstract: Code for visualizing the training history and smooth the line
'''
import os
import sys
import argparse
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def read_file(file_name):
    '''
    :param file_name: History file to be read
    :return A list
    '''
    if args.mode == 'train':
        file_data = []
        with open(file_name, 'r') as f:
            for line in f:
                data = line[:-1]
                data = float(data)
                data = round(data, 2)
                file_data.append(data)
        return file_data
    elif args.mode == 'test':
        file_data = []
        with open(file_name, 'r') as f:
            for line in f:
                data = line[:-1]
                file_data.append(data)
        return file_data

def data_preprocess(data):
    _data = []
    for line in data:
        line = line[1:-1]
        line = line.split(',')
        line = [float(num) for num in line]
        line = sorted(line)
        # line = line[1:]
        # line = np.mean(line)
        line = max(line)
        print (line)
        _data.append(line)
    return _data

 
def smooth(data):
    # tmp = scipy.signal.savgol_filter(data, 53, 3)
    tmp = signal.savgol_filter(data, 49, 3)
    return tmp

def plot_figure(loss, smooth_loss, acc, smooth_acc):
    fig = plt.figure(dpi=128, figsize=(10,6))
    plt.plot(loss, color='coral', alpha=0.2, label='Train Loss')
    plt.plot(smooth_loss,color='coral', label='Smoothed Loss')
    plt.plot(acc, color='royalblue', alpha=0.2, label='Train Accuracy')
    plt.plot(smooth_acc, color='royalblue', label='Smoothed Accuracy')
    plt.legend(loc='upper right')
    plt.title('{} {}-way {}-shot {} Process'.format(dataset, n_way, k_shot, args.mode))
    plt.xlabel('Meta Steps', fontsize=16)
    plt.ylabel('', fontsize=16)
    # plt.tick_params(axis='both', which='major', labelsize=16)
    plt.show()

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    # Dataset options
    argparse.add_argument('--dataset', type=str, help='Dataset miniimagenet or omniglot', default='miniimagenet')
    # Task options
    argparse.add_argument('--mode', type=str, help='Train process or test process', default='train')
    argparse.add_argument('--n_way', type=int, help='N-way', default=5)
    argparse.add_argument('--k_shot', type=int, help='K-shot', default=1)
    argparse.add_argument('--his_dir', type=str, help='Path to the training history directory', default='../../historys')
    # Generate args
    args = argparse.parse_args()
    
    dataset = args.dataset
    n_way = args.n_way
    k_shot = args.k_shot
    os.chdir(args.his_dir)
    if args.mode == 'train':
        loss = read_file('{}-{}-way-{}-shot-train.txt'.format(dataset, n_way, k_shot))
        acc = read_file('{}-{}-way-{}-shot-acc.txt'.format(dataset, n_way, k_shot))
    elif args.mode == 'test':
        loss = read_file('{}-{}-way-{}-shot-loss-test.txt'.format(dataset, n_way, k_shot))
        acc = read_file('{}-{}-way-{}-shot-acc-test.txt'.format(dataset, n_way, k_shot))
        # pre process
        loss = data_preprocess(loss)
        acc = data_preprocess(acc)

    smooth_loss = smooth(loss)
    smooth_acc = smooth(acc)

    plot_figure(loss, smooth_loss, acc, smooth_acc)