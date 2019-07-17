#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 12:51:57 2019
@author: shangranq
"""

from util import *
from model import _FCN, FCN_weights_init, CNN
from train_inference_util import *
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np
import torch.optim as optim
from random import shuffle
import os
import itertools
from data_loader import dataset_FCN, dataset_CNN
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(100)
batch_size = 6
lr = 0.0001
beta = 0.5
drop_rate = 0.6
epoches = 200
fil_num = 20

ADNI_dir = "/data/datasets/ADNI/"
NACC_dir = "/data/datasets/NACC/"
FHS_dir = "/data/datasets/FHS/"
AIBL_dir = "/data/datasets/AIBL/"
CNN_model_dir = "../CNN_model/"

# initialize model
Model = CNN(fil_num, drop_rate)
Model.apply(FCN_weights_init)
Model = Model.cuda()

# initialize optimizer and loss
optimizer = optim.Adam(Model.parameters(), lr=lr, betas=(beta, 0.999))
loss = nn.CrossEntropyLoss(weight=torch.Tensor([1.15, 1.51])).cuda()

# train FCN and save model
dataloader_ADNI_train = DataLoader(dataset_CNN(ADNI_dir, 'ADNI', 'train'), batch_size=batch_size, shuffle=True)
dataloader_ADNI_valid = DataLoader(dataset_CNN(ADNI_dir, 'ADNI', 'valid'), batch_size=batch_size, shuffle=True)
model_weight_epoch = train_FCN(epoches, FCN, dataloader_ADNI_train, dataloader_ADNI_valid, optimizer, loss, FCN_model_dir) 

# create all risk maps
state_dict_name = 'FCN_{}.pth'.format(fcn_epoch)
dataloader_ADNI = DataLoader(dataset_CNN(ADNI_dir, 'ADNI', 'inference'), batch_size=1, shuffle=False)
dataloader_NACC = DataLoader(dataset_CNN(NACC_dir, 'NACC', 'inference'), batch_size=1, shuffle=False)
dataloader_AIBL = DataLoader(dataset_CNN(AIBL_dir, 'AIBL', 'inference'), batch_size=1, shuffle=False)
dataloader_FHS = DataLoader(dataset_CNN(FHS_dir, 'FHS', 'inference'), batch_size=1, shuffle=False)
inference_CNN(FCN, FCN_model_dir, state_dict_name, dataloader_ADNI, 'ADNI', '../Risk_maps/ADNI/')
inference_CNN(FCN, FCN_model_dir, state_dict_name, dataloader_NACC, 'NACC', '../Risk_maps/NACC/')
inference_CNN(FCN, FCN_model_dir, state_dict_name, dataloader_AIBL, 'AIBL', '../Risk_maps/AIBL/')
inference_CNN(FCN, FCN_model_dir, state_dict_name, dataloader_FHS, 'FHS', '../Risk_maps/FHS/')

Heatmap('../Risk_maps/ADNI/', ADNI_dir, 'train')
Heatmap('../Risk_maps/ADNI/', ADNI_dir, 'valid')
Heatmap('../Risk_maps/ADNI/', ADNI_dir, 'test')
Heatmap('../Risk_maps/FHS/', FHS_dir, 'test')
Heatmap('../Risk_maps/NACC/', NACC_dir, 'test')
Heatmap('../Risk_maps/AIBL/', AIBL_dir, 'test')

