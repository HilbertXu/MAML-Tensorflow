'''
    Date: 27th Feb 2020
    Author: HilbertXu
    Abstract: A neural network created by purely python
'''
import pickle
import copy
import random
import numpy as np 
import matplotlib.pyplot as plt 
import argparse
from collections import defaultdict

# Create a special dictionary that returns 0 if the elements are not set
GradDict = lambda: defaultdict(lambda: 0)

# Normalize function
normalize = lambda x: (x - x.mean()) / (x.std() + 1e-8)

def build_weights(hidden_dim=40):
    '''
    :param hidden_dim: Number of Neure
    :return weights & bias for the forward pass
    '''
    h = hidden_dim
    d = {}
    