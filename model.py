from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import save_image
import torch.backends.cudnn as cudnn
import torch.nn.init as init
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.backends.cudnn as cudnn

import numpy as np
import math, pdb

class Simple(nn.Module):
    def __init__(self, w = 200, h = 66):
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
    def __init__(self, w = 200, h = 66):
        super().__init__()
        self.w = w
        self.h = h
        self.conv1 = nn.Conv2d(3,24,5,2)
        self.w, self.h = calc_out_size(self.w, self.h, 5,0,2)

        self.conv2 = nn.Conv2d(24, 36, 5, 2)
        self.conv2_bn = nn.BatchNorm2d(36)
        self.w, self.h = calc_out_size(self.w, self.h, 5,0,2)

        self.conv3 = nn.Conv2d(36, 48, 5, 2)
        self.conv3_bn = nn.BatchNorm2d(48)
        self.w, self.h = calc_out_size(self.w, self.h, 5,0,2)

        self.conv4 = nn.Conv2d(48, 64, 3, 1)
        self.conv4_bn = nn.Conv2d(64)
        self.w, self.h = calc_out_size(self.w, self.h, 3,0,1)

        self.conv5 = nn.Conv2d(64, 64, 3, 1)
        self.conv5_bn = nn.Conv2d(64)
        self.w, self.h = calc_out_size(self.w, self.h, 3,0,1)

        self.fc1 = nn.Linear(self.w*self.h*64, 1164)
        self.fc2 = nn.Linear(1164, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 10)
        self.fc5 = nn.Linear(10, 1)

        self.dropout = nn.Dropout()


    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2_bn(self.conv2(x)))
        x = F.elu(self.conv3_bn(self.conv3(x)))
        x = F.elu(self.conv4_bn(self.conv4(x)))
        x = F.elu(self.conv5_bn(self.conv5(x)))
        x = self.dropout(x)

        x = x.view(-1, self.w*self.h*64)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.dropout(x)
        x = F.elu(self.fc4(x))
        x = F.elu(self.fc5(x))
        return x

# retrieved from 
class CarModel(nn.Module):
    def __init__(self):
        super(CarModel, self).__init__()
        self.conv_layers = nn.Sequential(
            # input is batch_size x 3 x 66 x 200
            nn.Conv2d(3, 24, 5, stride=2, bias=False),
            #nn.ELU(0.2, inplace=True),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2, bias=False),
            nn.ELU(),
            nn.BatchNorm2d(36),

            nn.Conv2d(36, 48, 5, stride=2, bias=False),
            nn.ELU(),
            nn.BatchNorm2d(48),

            nn.Conv2d(48, 64, 3, stride=1, bias=False),
            nn.ELU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 3, stride=1, bias=False),
            nn.ELU(),
            nn.Dropout(p=0.4)
        )
        self.linear_layers = nn.Sequential(
            #input from sequential conv layers
            nn.Linear(in_features=64*1*18, out_features=100, bias=False),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=50, bias=False),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10, bias=False),
            nn.ELU(),
            nn.Linear(in_features=10, out_features=1, bias=False))
        self._initialize_weights()

    # custom weight initialization
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                init.normal(m.weight, mean=0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal(m.weight, mean=1, std=0.02)
                init.constant(m.bias, 0)

    def forward(self, input):
        output = self.conv_layers(input)
        output = output.view(output.size(0), 64*1*18)
        output = self.linear_layers(output)
        return output
