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


a = np.ones(5)

a = np.arange(24).reshape(4,6).astype(np.float)
b = torch.from_numpy(a)
print('b',b)
print('dim = 0',torch.mean(b, dim = 0))
print('dim = 1',torch.mean(b, dim = 1))


print('dim = 0', np.mean(a, axis= 0))
print('dim = 1', np.mean(a, axis= 1))
