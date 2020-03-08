'''
    Date: 6th Mar 2020
    Author: HilbertXu
    Abstract: Code for visualizing the training history and smooth the line
'''
import os
import sys
import argparse
import scipy.signal as signal
import matplotlib.pyplot as plt

def read_file(file_name):
    '''
    :param file_name: History file to be read
    :return A list
    '''
    file_data = []
    with open(file_name, 'r') as f:
        for line in f:
            data = line[:-1]
            data = float(data)
            data = round(data, 2)
            file_data.append(data)
    return file_data

def smooth(data):
    # tmp = scipy.signal.savgol_filter(data, 53, 3)
    tmp = signal.savgol_filter(data, 199, 3)
    return tmp

def plot_figure(loss, smooth_loss, acc, smooth_acc):
    fig = plt.figure(dpi=128, figsize=(10,6))
    plt.plot(loss, color='coral', alpha=0.2, label='Train Loss')
    plt.plot(smooth_loss,color='coral', label='Smoothed Loss')
    plt.plot(acc, color='royalblue', alpha=0.2, label='Train Accuracy')
    plt.plot(smooth_acc, color='royalblue', label='Smoothed Accuracy')
    plt.legend(loc='upper right')
    plt.title('{} {}-way {}-shot Training Process'.format(dataset, n_way, k_shot))
    plt.xlabel('Meta Steps', fontsize=16)
    plt.ylabel('', fontsize=16)
    # plt.tick_params(axis='both', which='major', labelsize=16)
    plt.show()

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    # Dataset options
    argparse.add_argument('--dataset', type=str, help='Dataset miniimagenet or omniglot', default='miniimagenet')
    # Task options
    argparse.add_argument('--n_way', type=int, help='N-way', default=5)
    argparse.add_argument('--k_shot', type=int, help='K-shot', default=1)
    argparse.add_argument('--his_dir', type=str, help='Path to the training history directory', default='../../historys/')
    # Generate args
    args = argparse.parse_args()
    
    dataset = args.dataset
    n_way = args.n_way
    k_shot = args.k_shot
    loss = read_file(args.his_dir + '/{}-{}-way-{}-shot-train.txt'.format(dataset, n_way, k_shot))
    acc = read_file(args.his_dir + '/{}-{}-way-{}-shot-acc.txt'.format(dataset, n_way, k_shot))
    smooth_loss = smooth(loss)
    smooth_acc = smooth(acc)

    plot_figure(loss, smooth_loss, acc, smooth_acc)