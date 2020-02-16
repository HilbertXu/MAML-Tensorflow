# -*- coding: UTF-8 -*-
import os
import sys
import random
import numpy as np
import tensorflow as tf 
import time
import matplotlib.pyplot as plt
from task_generator import generate_dataset
from meta_learner import MetaLearner

def copy_model(model, x):
    copied_model = create_model_via_keras()
    copied_model(x)
    copied_model.set_weights(model.get_weights())
    return copied_model