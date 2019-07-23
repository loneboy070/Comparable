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


x = torch.tensor([2., 2], requires_grad=True)
y = x**2 + x
z = y.sum()
z.backward()
print(x.grad)
with torch.no_grad():
    x = x+1

z.backward()
print(x.grad)