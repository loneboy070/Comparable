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

# import argparse
import config

def _generate_x_data(n_train, n_test, feature_dim):
    """
    Generate raw data x = (a,b,c,d,e) (standard normal distribution)
    """
    x_train = np.random.normal(size=[n_train, feature_dim])
    x_test = np.random.normal(size=[n_test, feature_dim])

    return x_train, x_test

def _generate_y_data(x_data):
    """Generate the y labels for given x_data
    y = (a^2+b^2+c^2)* (d+1) * (e)
    Return: y_data
    """
    temp = np.power(x_data[:,:3],2)
    temp = np.sum(temp, axis = 1)
    y_data = np.multiply(temp, x_data[:,2])
    y_data = np.multiply(y_data, x_data[:,3]+1)
    y_data = np.multiply(y_data, x_data[:,4])

    return y_data


if __name__ == '__main__':
    args = config.parser.parse_args()
    # print(args.N_TRAIN)
    N_TRAIN = args.N_TRAIN
    N_TEST = args.N_TEST
    FEATURE_DIM = args.FEATURE_DIM
    print(_generate_x_data(N_TRAIN, N_TEST, FEATURE_DIM))
