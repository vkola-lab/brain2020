from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import numpy as np
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

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate, kernel, pooling, BN=True, relu_type='leaky'):
        super().__init__()
        kernel_size, kernel_stride, kernel_padding = kernel
        pool_kernel, pool_stride, pool_padding = pooling
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, kernel_stride, kernel_padding, bias=False)
        self.pooling = nn.MaxPool3d(pool_kernel, pool_stride, pool_padding)
        self.BN = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU() if relu_type=='leaky' else nn.ReLU()
        self.dropout = nn.Dropout(drop_rate) 
       
    def forward(self, x):
        x = self.conv(x)
        x = self.pooling(x)
        x = self.BN(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class _CNN(nn.Module):
    def __init__(self, fil_num, drop_rate):
        super(_CNN, self).__init__()
        self.block1 = ConvLayer(1, fil_num, 0.1, (7, 2, 0), (3, 2, 0))
        self.block2 = ConvLayer(fil_num, 2*fil_num, 0.1, (4, 1, 0), (2, 2, 0))
        self.block3 = ConvLayer(2*fil_num, 4*fil_num, 0.1, (3, 1, 0), (2, 2, 0))
        self.block4 = ConvLayer(4*fil_num, 8*fil_num, 0.1, (3, 1, 0), (2, 1, 0))
        self.dense1 = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(8*fil_num*6*8*6, 30),
        )
        self.dense2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(30, 2),
        )

    def forward(self, x, stage='normal'):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.dense1(x)
        if stage == 'get_features':
            return x
        else:
            x = self.dense2(x)
            return x


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
        self.num = num

    def forward(self, x, stage='train'):
        x = self.features(x)
        if stage != 'inference':
            x = x.view(-1, self.feature_length)
        x = self.classifier(x)
        return x

    def dense_to_conv(self):
        fcn = copy.deepcopy(self)
        A = fcn.classifier[1].weight.view(30, 8*self.num, 6, 6, 6)
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


class _MLP_A(nn.Module):
    "MLP that only use DPMs from fcn"
    def __init__(self, in_size, drop_rate, fil_num):
        super(_MLP_A, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_size)   
        self.bn2 = nn.BatchNorm1d(fil_num)  
        self.fc1 = nn.Linear(in_size, fil_num)
        self.fc2 = nn.Linear(fil_num, 2)
        self.do1 = nn.Dropout(drop_rate)
        self.do2 = nn.Dropout(drop_rate) 
        self.ac1 = nn.LeakyReLU()
    
    def forward(self, X):
        X = self.bn1(X)
        out = self.do1(X)
        out = self.fc1(out)
        out = self.bn2(out)
        out = self.ac1(out)
        out = self.do2(out)
        out = self.fc2(out)
        return out


class _MLP_B(nn.Module):
    "MLP that only use age gender MMSE"
    def __init__(self, in_size, drop_rate, fil_num):
        super(_MLP_B, self).__init__()        
        self.fc1 = nn.Linear(in_size, fil_num)
        self.fc2 = nn.Linear(fil_num, 2)
        self.do1 = nn.Dropout(drop_rate)
        self.do2 = nn.Dropout(drop_rate) 
        self.ac1 = nn.LeakyReLU()
    
    def forward(self, X):
        out = self.do1(X)
        out = self.fc1(out)
        out = self.ac1(out)
        out = self.do2(out)
        out = self.fc2(out)
        return out


class _MLP_C(nn.Module):
    "MLP that use DPMs from fcn and age, gender and MMSE"
    def __init__(self, in_size, drop_rate, fil_num):
        super(_MLP_C, self).__init__()        
        self.fc1 = nn.Linear(in_size, fil_num)
        self.fc2 = nn.Linear(fil_num, 2)
        self.do1 = nn.Dropout(drop_rate)
        self.do2 = nn.Dropout(drop_rate) 
        self.ac1 = nn.LeakyReLU()
    
    def forward(self, X1, X2):
        X = torch.cat((X1, X2), 1)
        out = self.do1(X)
        out = self.fc1(out)
        out = self.ac1(out)
        out = self.do2(out)
        out = self.fc2(out)
        return out


class _MLP_D(nn.Module):
    "MLP that use cnn features and age, gender and MMSE"
    def __init__(self, in_size, drop_rate, fil_num):
        super(_MLP_D, self).__init__()
        self.fc1 = nn.Linear(in_size, fil_num)
        self.fc2 = nn.Linear(fil_num, 2)
        self.do1 = nn.Dropout(drop_rate)
        self.do2 = nn.Dropout(drop_rate)
        self.ac1 = nn.LeakyReLU()

    def forward(self, X1, X2):
        X = torch.cat((X1, X2), 1)
        out = self.do1(X)
        out = self.fc1(out)
        out = self.ac1(out)
        out = self.do2(out)
        out = self.fc2(out)
        return out
