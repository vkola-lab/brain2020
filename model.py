from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
from utils import read_txt
import random
import copy


"""
models are defined in this scripts:

    1. FCN
        (a). details

    2. MLP 
        (a). details

    3. CNN 
        (a). details
        
"""

class _FCN(nn.Module):
    def __init__(self, num, p):
        super(_FCN, self).__init__()
        self.features = nn.Sequential(
            # 47, 47, 47
            nn.Conv3d(1, num, 4, 1, 0, bias=False),
            nn.MaxPool3d(2, 1, 0),
            nn.BatchNorm3d(num),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            # 43, 43, 43
            nn.Conv3d(num, 2*num, 4, 1, 0, bias=False),
            nn.MaxPool3d(2, 2, 0),
            nn.BatchNorm3d(2*num),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            # 20, 20, 20
            nn.Conv3d(2*num, 4*num, 3, 1, 0, bias=False),
            nn.MaxPool3d(2, 2, 0),
            nn.BatchNorm3d(4*num),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            # 9, 9, 9
            nn.Conv3d(4*num, 8*num, 3, 1, 0, bias=False),
            nn.MaxPool3d(2, 1, 0),
            nn.BatchNorm3d(8*num),
            nn.LeakyReLU(),
            # 6, 6, 6
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p),
            nn.Linear(8*num*6*6*6, 30),
            nn.LeakyReLU(),
            nn.Dropout(p),
            nn.Linear(30, 2),
        )
        self.feature_length = 8*num*6*6*6

    def forward(self, x, stage='train'):
        """
        :param x:
        :param stage: why need the stage?
        """
        x = self.features(x)
        if stage != 'inference':
            x = x.view(-1, self.feature_length)
        x = self.classifier(x)
        return x

    def dense_to_conv(self):
        fcn = copy.deepcopy(self)
        A = fcn.classifier[1].weight.view(30, 160, 6, 6, 6)
        B = fcn.classifier[4].weight.view(2, 30, 1, 1, 1)
        C = fcn.classifier[1].bias
        D = fcn.classifier[4].bias
        fcn.classifier[1] = nn.Conv3d(160, 30, 6, 1, 0).cuda()
        fcn.classifier[4] = nn.Conv3d(30, 2, 1, 1, 0).cuda()
        fcn.classifier[1].weight = nn.Parameter(A)
        fcn.classifier[4].weight = nn.Parameter(B)
        fcn.classifier[1].bias = nn.Parameter(C)
        fcn.classifier[4].bias = nn.Parameter(D)
        return fcn



class _CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3, 1, 0),
            nn.ReLU(),
            nn.Conv3d(8, 8, 3, 1, 0),
            nn.ReLU(),
            nn.MaxPool3d(2, 2, 0),

            nn.Conv3d(8, 16, 3, 1, 0),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3, 1, 0),
            nn.ReLU(),
            nn.MaxPool3d(2, 2, 0),

            nn.Conv3d(16, 32, 3, 1, 0),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, 1, 0),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, 1, 0),
            nn.ReLU(),
            nn.MaxPool3d(2, 2, 0),

            nn.Conv3d(32, 64, 3, 1, 0),
            nn.ReLU(),
            nn.Conv3d(64, 64, 3, 1, 0),
            nn.ReLU(),
            nn.Conv3d(64, 64, 3, 1, 0),
            nn.ReLU(),
            nn.MaxPool3d(2, 2, 0),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64*6*6*8, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.7),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 64*36*8)
        x = self.classifier(x)
        return x


class _MLP(nn.Module):