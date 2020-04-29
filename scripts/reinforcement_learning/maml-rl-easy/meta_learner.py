import numpy as np
import tensorflow as tf
from collections import OrderedDict

class BaseMetaLearner(object):

    def inner_loss(self, episodes, params=None):
        raise NotImplementedError

    def surrogate_loss(self, episodes, old_pis=None):
        raise NotImplementedError

    def adapt(self, episodes, first_order=False):
        raise NotImplementedError

    def step(self, episodes):
        raise NotImplementedError

class MetaPolicyGradient(BaseMetaLearner):
    def __init__(
        self,
        policy,
        sampler,
        optimizer,
        gamma=0.95,
        inner_lr=0.01,
        outer_lr=0.001,
        ):
        self.policy = policy
        self.sampler = sampler
        self.optimizer = optimizer
        self.gamma = gamma
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr

        self.ep_obs = []
        self.ep_rs = []
        self.ep_acts = []  
    
    def inner_loss(self, episodes, params=None):
        return 0
 

