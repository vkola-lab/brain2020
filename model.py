#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 12:51:57 2019
@author: Shangranq
"""

import torch
import torch.nn as nn
import copy
import numpy as np

# define the FCN
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


def FCN_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class CNN(nn.Module):
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

