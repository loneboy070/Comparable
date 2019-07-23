
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

import config
args = config.parser.parse_args()

# import comparing 
import data_generator as dg


def debug_data_generation(func):
    """Check the dimension of the generated the data structures
    """
    x = np.array([[1,1,1,1,1], [0,1,0, -1, 1]])
    y = np.array([0.6, 0.0])
    x = x.astype(np.float)
    y = y.astype(np.float)

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    
    if torch.sum((func(x) - y).pow(2)):
        print('error occurs x and y, func(x)', x, y, func(x))

    return None

def _check_x_shape(X_TRAIN, X_TEST):
    """Check the shape of the matrix
    """

    if not np.shape(X_TRAIN)[0] ==  args.N_TRAIN and np.shape(X_TRAIN)[1] ==  args.FEATURE_DIM and np.shape(X_TEST)[0] == args.N_TEST and np.shape(X_TEST)[1] == args.FEATURE_DIM:
        print('x_data is wrong, X_TRAIN, X_TEST', np.shape(X_TRAIN), np.shape(X_TEST))

    return None


def debug_data_comination(func, tran_num, test_num):
    """Check whether the training data and the test data is combined corectly.
    """
    DIM1 = 3
    DIM2 = 4
    x_train = torch.ones([100,DIM1,DIM2])
    x_test = torch.zeros([30, DIM1, DIM2])
    
    y = torch.ones([tran_num+test_num, DIM1, DIM2])

    y[-test_num:] = 0
    if y != func(x_train,x_test):
        print('error for combinating the training data and test data')

    return None

if __name__ == '__main__':
    debug_data_generation(dg.generate_y_data)
    x_train, x_test = dg.generate_x_data(10, 5, 5)
    _check_x_shape(x_train,x_test)




