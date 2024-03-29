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

        self.Node_Sizes = Node_Sizes
        Node_Sizes_s = cp.deepcopy(Node_Sizes)
        Node_Sizes_g = cp.deepcopy(Node_Sizes)
        self.SUBNODE = args.SUBNODE

        print('Node_Sizes',Node_Sizes)
        if args.ENN:
            for i in range(len(Node_Sizes)):
                if i>0 and i < len(Node_Sizes)-1:
                    Node_Sizes_s[i] -= self.SUBNODE
                    Node_Sizes_g[i] = self.SUBNODE

        print('Node_Sizes_s',Node_Sizes_s)
        print('Node_Sizes_g',Node_Sizes_g)


        self.layers_s = self._generate_layers(Node_Sizes_s, bias=True)
        self.layers_g = self._generate_layers(Node_Sizes_g, bias=True)
        self.batchnorms = self._generate_batch_norms(Node_Sizes)
        self.dropout = self._generate_dropout()
        # print('Node_Sizes', Node_Sizes)

        self.parameters = nn.ModuleList(self.layers_g + self.layers_s)
        self.Node_Sizes = args.Node_Sizes
        # print(self.SUBNODE)
        return None

    def _generate_dropout(self):
        """Generate the dropout function
        """
        dropout = nn.Dropout(p=args.DROPOUT)
        return dropout

    def _generate_layers(self, Node_Sizes, bias = True):
        """Generate layers
        """
        layers = [None for _ in range(len(Node_Sizes)-1)]
        for i in range(len(Node_Sizes)-1):
            layers[i] = nn.Linear(self.Node_Sizes[i], Node_Sizes[i+1], bias = bias)
            # self.batchnorms[i] = nn.BatchNorm1d(Node_Sizes[i])
        return layers


    def _generate_batch_norms(self,  Node_Sizes):
        """Generate batch norm parameters
        """
        batchnorms = [None for _ in range(len(Node_Sizes))]
        for i in range(len(Node_Sizes)):
            batchnorms[i] = nn.BatchNorm1d(Node_Sizes[i])

        return batchnorms


    def forward(self, init_layer):
        """
        Generate neural network: 
        IF args.ENN = True:, shared network,
        if not, separate network
        """
        layer = init_layer

        for i in range(len(self.Node_Sizes)-1):
            prev_layer = layer
            # print('prev_layer',prev_layer)

            layer = self.layers_s[i](prev_layer)
            # print('prev_layer first',prev_layer)

            # print('raw layer',layer)

            if i != len(self.Node_Sizes)-2:
                if args.ENN:
                    layer_g = -torch.mul(self.layers_g[i](prev_layer),1/(args.N_ITEM))

                    layer_g += self.layers_g[i](torch.mean(prev_layer, dim=0))
                    
                    layer = torch.cat([layer,layer_g], dim=1)

                # print('layer', layer)
                layer = layer.clamp(min=0)
                if args.DROPOUT>0:
                    layer = self.dropout(layer)
 

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
