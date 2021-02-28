import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pdb

class Simple(nn.Module):
    def __init__(self, w, h):
        super().__init__()
        self.w = w
        self.h = h
        self.conv1 = nn.Conv2d(3,30,3,1)
        self.calc_out_size(3,0,1)

        self.conv2 = nn.Conv2d(30, 15, 3, 1)
        self.calc_out_size(3,0,1)
        self.calc_pool_size(2,2)

        self.conv3 = nn.Conv2d(15, 10, 3, 1)
        self.calc_out_size(3,0,1)
        self.conv3_bn = nn.BatchNorm2d(10)

        self.conv4 = nn.Conv2d(10, 5, 3, 1)
        self.calc_out_size(3,0,1)
        self.calc_pool_size(2,2)

        self.fc1 = nn.Linear(self.w*self.h*5, 500)
        self.fc1_bn = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 250)
        self.fc3 = nn.Linear(250, 100)
        self.fc3_bn = nn.BatchNorm1d(100)
        self.fc4 = nn.Linear(100, 25)
        self.fc5 = nn.Linear(25, 1)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2,2)
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x,2,2)
        # pdb.set_trace()
        x = x.view(-1, self.w*self.h*5)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3_bn(self.fc3(x)))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return x


    def calc_out_size(self, k_size, pooling, stride):
        self.w = int((self.w - k_size + 2*pooling)/stride + 1)
        self.h = int((self.h - k_size + 2*pooling)/stride + 1)

    def calc_pool_size(self, k_size, stride):
        self.w = int((self.w - k_size)/stride + 1)
        self.h = int((self.h - k_size)/stride + 1)




class Perf(nn.Module):
    pass


class WithSpeed(nn.Module):
    pass
