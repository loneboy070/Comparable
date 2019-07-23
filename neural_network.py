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


class EquiNetwork(torch.nn.Module):
    def __init__(self, Node_Sizes):
        """
        Equivariant Neural network
        """
        super(EquiNetwork, self).__init__()

        self.layers_s = self._generate_layers(Node_Sizes, bias=True)
        self.layers_g = self._generate_layers(Node_Sizes, bias=False)
        # print('Node_Sizes', Node_Sizes)

        self.parameters = nn.ModuleList(self.layers_g + self.layers_s)
        self.Node_Sizes = args.Node_Sizes

        return None

    def _generate_layers(self, Node_Sizes, bias = True):
        """Generate layers
        """
        layers = [None for _ in range(len(Node_Sizes)-1)]
        for i in range(len(Node_Sizes)-1):
            layers[i] = nn.Linear(Node_Sizes[i], Node_Sizes[i+1], bias = bias)
        
        return layers

    def forward(self, init_layer):
        """
        Generate neural network: 
        IF args.ENN = True:, shared network,
        if not, separate network
        """
        # init_layer = torch.from_numpy(init_layer)
        # next_layer = init_layer.float()
        layer = init_layer

        for i in range(len(self.Node_Sizes)-1):
            prev_layer = layer
            # print('prev_layer i',i, prev_layer)
            # print('self.layers_s[i],i ',i,  self.layers_s[i])
            layer = self.layers_s[i](prev_layer)
            if args.ENN:
                layer += self.layers_g[i](torch.mean(prev_layer, dim=0))
                # layer += self.layers_g[i](torch.max(prev_layer, dim=0)[0])

                
            if i != len(self.Node_Sizes)-2:
                layer = layer.clamp(min=0)       

        return layer

    

if __name__ == '__main__':
    Node_Sizes = args.Node_Sizes
    model = EquiNetwork(Node_Sizes)

    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for t in range(5000):
        # 순전파 단계: 모델에 x를 전달하여 예상하는 y 값을 계산합니다.
        batches = np.random.choice(N, size=3, replace=False)
        x_train = x[batches]

        y_pred = model(x_train)
        loss = criterion(y_pred, y[batches])

        # 변화도를 0으로 만들고, 역전파 단계를 수행하고, 가중치를 갱신합니다.
        # print(t,loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
