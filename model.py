from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import math, pdb

class Simple(nn.Module):
    def __init__(self, w, h):
        super().__init__()
        self.w = w
        self.h = h
        self.conv1 = nn.Conv2d(3,60,3,1)
        self.w, self.h = calc_out_size(self.w, self.h, 3,0,1)

        self.conv2 = nn.Conv2d(60, 30, 3, 1)
        self.w, self.h = calc_out_size(self.w, self.h, 3,0,1)
        self.w, self.h = calc_pool_size(self.w, self.h, 2,2)

        self.conv3 = nn.Conv2d(30, 10, 3, 1)
        self.w, self.h = calc_out_size(self.w, self.h, 3,0,1)
        self.conv3_bn = nn.BatchNorm2d(10)

        self.conv4 = nn.Conv2d(10, 5, 3, 1)
        self.w, self.h = calc_out_size(self.w, self.h, 3,0,1)
        self.w, self.h = calc_pool_size(self.w, self.h, 2,2)

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
        x = x.view(-1, self.w*self.h*5)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3_bn(self.fc3(x)))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return x


class Nvidia(nn.Module):
    def __init__(self, w, h):
        super().__init__()
        self.w = w
        self.h = h
        self.conv1 = nn.Conv2d(3,24,5,2)
        self.w, self.h = calc_out_size(self.w, self.h, 5,0,2)

        self.conv2 = nn.Conv2d(24, 36, 5, 2)
        self.w, self.h = calc_out_size(self.w, self.h, 5,0,2)
    
        self.conv3 = nn.Conv2d(36, 48, 5, 2)
        self.w, self.h = calc_out_size(self.w, self.h, 5,0,2)
        
        self.conv4 = nn.Conv2d(48, 64, 3, 1)
        self.w, self.h = calc_out_size(self.w, self.h, 3,0,1)

        self.conv5 = nn.Conv2d(64, 64, 3, 1)
        self.w, self.h = calc_out_size(self.w, self.h, 3,0,1)

        self.fc1 = nn.Linear(self.w*self.h*64, 1164)
        self.fc2 = nn.Linear(1164, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 10)
        self.fc5 = nn.Linear(10, 1) 


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        
        x = x.view(-1, self.w*self.h*64)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return x
