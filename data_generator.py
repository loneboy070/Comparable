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

def generate_x_data(n_train, n_test, feature_dim):
    """
    Generate raw data x = (a,b,c,d,e) (standard normal distribution)
    """
    torch.manual_seed(0)
    x_train = torch.randn(size=[n_train, feature_dim])
    x_test = torch.randn(size=[n_test, feature_dim])


    return x_train, x_test

def generate_y_data(x_data):
    """Generate the y labels for given x_data
    y = (a^2+b^2+c^2)* (d+1) * (e)/10.0
    Return: y_data
    """
    temp = x_data[:,:3].pow(2)
    temp = torch.sum(temp, dim = 1)

    # # test for simple y
    # y_data = temp

    # hard y
    y_data = torch.mul(temp, x_data[:,2])
    y_data = torch.mul(y_data, x_data[:,3]+1)
    y_data = torch.mul(y_data, x_data[:,4])
    y_data = y_data/10.0

    return y_data




if __name__ == '__main__':
    args = config.parser.parse_args()
    # print(args.N_TRAIN)
    N_TRAIN = args.N_TRAIN
    N_TEST = args.N_TEST
    FEATURE_DIM = args.FEATURE_DIM
    print(generate_x_data(N_TRAIN, N_TEST, FEATURE_DIM))
