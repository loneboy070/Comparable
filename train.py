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
# from neural_network1 import EquiNetwork
# from neural_network2 import EquiNetwork
from neural_network import EquiNetwork


import config
args = config.parser.parse_args()


class Train():
    def __init__(self):
        """
        Train the whole process
        """
        super(Train, self).__init__()
        self._init_parameters()
        self.generate_path()
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
        self.ITERATIONS = args.ITERATIONS
        self.START_ITER = args.START_ITER
        self.END_ITER = args.END_ITER
        self.LOSS_PRINT = args.LOSS_PRINT
        self.TEST_LOSS_PRINT = args.TEST_LOSS_PRINT
        self.TEST_BATCH = args.TEST_BATCH
        
        return None

    def generate_data(self):
        """Generate the data train data and test data
        """
        self.x_train, self.x_test = dg.generate_x_data(self.N_TRAIN, self.N_TEST, self.FEATURE_DIM)
        self.y_train, self.y_test = dg.generate_y_data(self.x_train), dg.generate_y_data(self.x_test)

        return None

    def generate_model(self):
        """Design the model for pytorch
        """
        self.model = EquiNetwork(self.Node_Sizes)
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.criterion_test = torch.nn.MSELoss(reduction='mean')
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.LR)

        return None

    def _backpropagation(self, batch):
        """Learning the neural network with prorper sizes of batch for a step
        """
        self.model.train()
        # self.model.eval()

        # 순전파 단계: 모델에 x를 전달하여 예상하는 y 값을 계산합니다.
        # batch = np.random.choice(self.N_TRAIN, size=self.N_ITEM, replace=False)
        x_train = self.x_train[batch]
        y_train = self.y_train[batch].view([-1, 1])
        y_pred = self.model(x_train)

        loss = self.criterion(y_pred, y_train)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        new_pred = self.model(x_train)

        return loss.item()

    def _get_batches(self):
        """Generate the batches with ENNSIZE * N_TRAIN/ENNSIZE
        Some samples may duplicated 
        """
        raw_string = np.arange(self.N_TRAIN)

        raw_string = np.append(raw_string, np.random.choice(raw_string,size=(self.N_TRAIN%self.N_ITEM),replace=False))
        # raw_string.append()

        np.random.shuffle(raw_string)
        batches = np.reshape(raw_string, [self.N_ITEM, -1])
        
        return batches


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
       
                error_ratio = 1
                mix_accuracy_mat[test_index] += error_ratio

                mix_pred_mat[test_index] += error_ratio * y_mix_pred[-1].item()

        mix_pred_mat = mix_pred_mat/self.TEST_BATCH

        test_loss = self.criterion_test(self.y_test.view(mix_pred_mat.size()), mix_pred_mat).item()

        return test_loss

    def generate_path(self):
        """Generate the pathes.
        """
        self.weight_path='./saved_model/weight/'
        self.log_path='./logs/'
        self.name ='_ITERATIONS_'+str(self.ITERATIONS)+'_train_'+str(self.N_TRAIN)+"_ITEM_"+str(self.N_ITEM)+'_BATCH_'+str(self.TEST_BATCH)+'_ENN_'+str(args.ENN)+'_SUBNODE_'+str(args.SUBNODE)+'_SEED_DATA_'+str(args.SEED_DATA)+'_SEED_TRAIN_'+str(args.SEED_TRAIN)
        if not (os.path.isdir(self.weight_path)):
            os.makedirs(os.path.join(self.weight_path))
        if not (os.path.isdir(self.log_path)):
            os.makedirs(os.path.join(self.log_path))
        return None


    def training(self):
        """ Traing the neural network during iterations.
        """
        # path='./saved_model/weight'
        # path+='_ITERATIONS_'+str(self.ITERATIONS)+'_ENN_'+str(args.ENN)+'_train_'+str(self.N_TRAIN)+"_ITEM_"+str(self.N_ITEM)+'_BATCH_'+str(self.TEST_BATCH)+'_SEED_'+str(args.SEED_DATA)
        # path+='.pkl'
        if self.START_ITER:
            # print('ha NODE', args.SUBNODE)
            # self.model.eval()
            # print('before, self.model(np.ones(4,5))', self.model(torch.ones([4,5])))
            self.load_model()            
            # print('after, self.model(np.ones(4,5))', self.model(torch.ones([4,5])))
            # exit(0)

        time1 = time.time()

        train_loss = 0

        for t in range(self.START_ITER, self.END_ITER):
            batches = self._get_batches()
            for batch in batches:
                train_loss += self._backpropagation(batch)

            if t % self.LOSS_PRINT == 0 and t>self.START_ITER:
                # print('                                               Training loss for iteration t:', t, train_loss/np.float(self.LOSS_PRINT))
                result_str = '\n Training loss for iteration t: ' +str(t) + "   "+str(train_loss/np.float(self.LOSS_PRINT))
                train_loss = 0

                if self.TEST_LOSS_PRINT:
                    test_loss = self._testing()
                    result_str += '      Test loss for iteration t:' + str(int(t)) + "  "+ str(test_loss) 

                print(result_str)
                

                txtfile = open(self.log_path+self.name+'.txt',"a") 
                txtfile.write(result_str)
                txtfile.close()

        # torch.save(self.model.state_dict(), path)
        torch.save(self.model.state_dict(), self.weight_path+self.name+'.pkl')
        time2 = time.time()

        print("running successfully ends within "+str(time2-time1))

        return None

    def load_model(self):
        """Load the model with the path
        """

        self.model.load_state_dict(torch.load(self.weight_path+self.name+'.pkl'))

        return None


    

if __name__ == '__main__':
    print('args.ENN',args.ENN, type(args.ENN))
    print('args.LOSS_PRINT',args.LOSS_PRINT, type(args.LOSS_PRINT))
    print('args.LR',args.LR, type(args.LR))
    Trainer = Train()
    Trainer.generate_data()
    Trainer.generate_model()
    Trainer.training()

            
            





    
