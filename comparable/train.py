import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy as cp

import data_generator as dg

import config
args = config.parser.parse_args()


class Train():
    def __init__(self):
        """
        Train the whole process
        """
        super(Train, self).__init__()



    def _init_parameters(self):
        """Initialize the paramters
        """
        self.N_TRAIN = args.N_TRAIN
        self.N_TEST = args.N_TEST 
        self.FEATURE_DIM = args.FEATURE_DIM
        
        return None

    def _generate_data(self):
        """Generate the data train data and test data
        """
        self.x_train, self.x_test = dg._generate_x_data(self.N_TRAIN, self.N_TEST, self.FEATURE_DIM)
        self.y_train, self.y_test = dg._generate_y_data(self.x_train), dg._generate_y_data(self.x_test)

        return None

    
