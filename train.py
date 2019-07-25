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
from neural_network import EquiNetwork
# from neural_network2 import EquiNetwork


import config
args = config.parser.parse_args()


class Train():
    def __init__(self):
        """
        Train the whole process
        """
        super(Train, self).__init__()
        self._init_parameters()
        # self.generate_data()

        return None

    def _init_parameters(self):
        """Initialize the paramters
        """
        self.N_TRAIN = args.N_TRAIN
        self.N_TEST = args.N_TEST 
        self.FEATURE_DIM = args.FEATURE_DIM
        self.Node_Sizes = args.Node_Sizes
        self.N_ITEM = args.N_ITEM
        self.STEPS = args.STEPS
        self.LOSS_PRINT = args.LOSS_PRINT
        self.TEST_LOSS_PRINT = args.TEST_LOSS_PRINT
        self.TEST_BATCH = args.TEST_BATCH
        
        return None

    def generate_data(self):
        """Generate the data train data and test data
        """
        self.x_train, self.x_test = dg.generate_x_data(self.N_TRAIN, self.N_TEST, self.FEATURE_DIM)
        self.y_train, self.y_test = dg.generate_y_data(self.x_train), dg.generate_y_data(self.x_test)

        # print('self.x_train',self.x_train)
        # print('self.x_test',self.x_test)
        # print('self.y_train',self.y_train)
        # print('self.y_test',self.y_test)
        
        # self.y_train += 0.2 * torch.randn(size=self.y_train.size())

        return None


    def generate_model(self):
        """Design the model for pytorch
        """
        self.model = EquiNetwork(self.Node_Sizes)

        self.criterion = torch.nn.MSELoss(reduction='mean')
        # self.criterion = torch.nn.MSELoss(reduction='sum')

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

        return None


    def _backpropagation(self):
        """Learning the neural network with prorper sizes of batches for a step
        """
        self.model.train()
        # self.model.eval()

        # 순전파 단계: 모델에 x를 전달하여 예상하는 y 값을 계산합니다.
        batches = np.random.choice(self.N_TRAIN, size=self.N_ITEM, replace=False)
        x_train = self.x_train[batches]
        y_train = self.y_train[batches].view([-1, 1])

        y_pred = self.model(x_train)

        loss = self.criterion(y_pred, y_train)
        # print('x_train',x_train)
        # print('y_pred', y_pred)
        # print('y_train', y_train)
        # print('loss', loss)
        # exit(0)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        new_pred = self.model(x_train)

        return loss.item()

    def _testing(self):
        """Test the neural network with nongiven samples.
        """
        self.model.eval()
    
        test_loss_sum = 0

        mix_pred_mat = torch.zeros(self.N_TEST)
        mix_accuracy_mat = torch.zeros(self.N_TEST)


        for mix_batch in range(self.TEST_BATCH):

            train_batches =  np.random.choice(self.N_TRAIN, size=self.N_ITEM-1, replace=False)
            x_train = self.x_train[train_batches]
            y_train = self.y_train[train_batches]

            for test_index in range(self.N_TEST):
                
                x_test = self.x_test[[test_index]]
                y_test = self.y_test[[test_index]]


                x_mix = torch.cat((x_train, x_test))
                y_mix = torch.cat((y_train, y_test))
                y_mix = y_mix.view([-1,1])

                y_mix_pred = self.model(x_mix)
                # print('y_mix_pred',y_mix_pred)
                # print('mix_pred_mat[test_index]', mix_pred_mat)
                # if test_index == 0:
                #     print('x_test', x_test)
                    # print('predict result', y_mix_pred[-1].item())
                # error_train = self.criterion(y_mix_pred[:-1],y_train.view(y_mix_pred[:-1].size()))
                
                # error_ratio = torch.exp(-error_train)
                error_ratio = 1
                mix_accuracy_mat[test_index] += error_ratio

                mix_pred_mat[test_index] += error_ratio * y_mix_pred[-1].item()
        
        # # mix_pred_mat = mix_pred_mat/self.TEST_BATCH
        # print('mix_accuracy_mat',mix_accuracy_mat)
        # mix_pred_mat = torch.mul(mix_pred_mat, mix_accuracy_mat.pow(-1))
        mix_pred_mat = mix_pred_mat/self.TEST_BATCH

        test_loss = self.criterion(self.y_test.view(mix_pred_mat.size()), mix_pred_mat).item()

        # test_loss = test_loss_sum/self.N_TEST 
        
        return test_loss


    def training(self):
        path='./saved_model/weight'
        """ Traing the neural network during iterations.
        """
        path+='STEPS_'+str(self.STEPS)+'ENN_'+str(args.ENN)+'train_'+str(self.N_TRAIN)+"ITEM_"+str(self.N_ITEM)+'BATCH_'+str(self.TEST_BATCH)+'SEED_'+str(args.SEED_DATA)
        path+='.pkl'
        train_loss = 0 
        for t in range(self.STEPS):
            train_loss += self._backpropagation()
            if t % self.LOSS_PRINT == 0 and t>0:
                # print('                                               Training loss for iteration t:', t, train_loss/np.float(self.LOSS_PRINT))
                result_str = 'Training loss for iteration t: ' +str(t) + "   "+str(train_loss/np.float(self.LOSS_PRINT))
                train_loss = 0

                if self.TEST_LOSS_PRINT:
                    test_loss = self._testing()
                    result_str += '      Test loss for iteration t:' + str(int(t/1000)) + "  "+ str(test_loss)

                print(result_str)

        torch.save(self.model.state_dict(), path)



        return None

if __name__ == '__main__':
    Trainer = Train()
    Trainer.generate_data()
    Trainer.generate_model()
    Trainer.training()
            
            





    
