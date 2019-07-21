#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 12:51:57 2019
@author: Shangranq
"""

from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
from util import load_txt, padding
from random import shuffle
import random

class dataset_FCN(Dataset):

    def __init__(self, Data_dir, dset_name, stage):
        self.Data_dir = Data_dir
        self.stage = stage
        self.Data_list = load_txt(Data_dir, '{}_Data.txt'.format(dset_name))
        self.Label_list = load_txt(Data_dir, '{}_Label.txt'.format(dset_name))
        if dset_name == 'ADNI':
            if self.stage == 'train':
                self.index_list = [i for i in range(0, 257)]
            elif self.stage == 'valid':
                self.index_list = [i for i in range(257, 337)]
            elif self.stage == 'test':
                self.index_list = [i for i in range(337, 417)] 
            elif self.stage == 'inference':
                self.index_list = [i for i in range(417)]
        else:
            self.index_list = [i for i in range(len(self.Data_list))]

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        index = self.index_list[idx]
        data = np.load(self.Data_dir + self.Data_list[index])
        label = 0 if self.Label_list[index]=='NL' else 1
        if self.stage == 'train':
            x, y, z = random.randint(0, 134), random.randint(0, 170), random.randint(0, 134)
            patch = data[x:x+47, y:y+47, z:z+47]
            return np.expand_dims(patch, axis=0), label
        elif self.stage == 'valid':
            array_list = []
            patch_locs = [[25, 90, 30], [115, 90, 30], [67, 90, 90], [67, 45, 60], [67, 135, 60]]
            for i, loc in enumerate(patch_locs):
                x, y, z = loc
                patch = data[x:x+47, y:y+47, z:z+47]
                array_list.append(np.expand_dims(patch, axis = 0))
            data = Variable(torch.FloatTensor(np.stack(array_list, axis = 0)))
            label = Variable(torch.LongTensor([label]*5))
            return data, label
        else:
            return np.expand_dims(padding(data), axis=0), label


class dataset_CNN(dataset_FCN):
    def __getitem__(self, idx):
        index = self.index_list[idx]
        data = np.load(self.Data_dir + self.Data_list[index]).astype(np.float32)
        if self.Label_list[index]=='NL':
            label = 0
        elif self.Label_list[index]=='AD':
            label = 1
        return np.expand_dims(data, axis=0), label


class dataset_longi_ADNI(Dataset):
    def __init__(self, Data_dir, stage):
        self.Data_dir = Data_dir
        self.stage = stage
        self.Data_list = load_txt(Data_dir, 'ADNI_Data.txt')
        self.Label_list = load_txt(Data_dir, 'ADNI_Label.txt')
        self.Rid_list = [int(a[11:15]) for a in self.Data_list]
        self.conv_rid = [22, 210, 232, 467, 548, 717, 779, 843, 883, 899, 1009] + \
                    [1063, 1194, 1200, 1202, 1203, 1241]
        self.conv_index = [self.Rid_list.index(i) for i in self.conv_rid]
        with open('./Longi_label.txt', 'w') as f:
            for i in range(417):
                if self.Label_list[i] == 'AD':
                    f.write('.\n')
                else:
                    if i in self.conv_index:
                        f.write('1\n')
                    else:
                        f.write('0\n')

        if self.stage == 'train':
            self.con_index = [i for i in self.conv_index if i < 257]
            self.non_index = [i for i in range(0, 257) if self.Label_list[i]=='NL'
                            and self.Rid_list[i] not in self.conv_rid]
        elif self.stage == 'valid':
            self.con_index = [i for i in self.conv_index if i >= 257 and i < 337]
            self.non_index = [i for i in range(257, 337) if self.Label_list[i]=='NL'
                            and self.Rid_list[i] not in self.conv_rid]
        elif self.stage == 'test':
            self.con_index = [i for i in self.conv_index if i >= 337]
            self.non_index = [i for i in range(337, 417) if self.Label_list[i]=='NL'
                            and self.Rid_list[i] not in self.conv_rid]
        elif self.stage == 'inference':
            self.con_index = []
            self.non_index = [i for i in range(0, 417) if self.Label_list[i]=='NL']
        self.index_list = self.con_index + self.non_index
        shuffle(self.index_list)
        
    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        index = self.index_list[idx]
        data = np.load(self.Data_dir + self.Data_list[index])
        label = 0 if index in self.non_index else 1
        if self.stage == 'train':
            x, y, z = random.randint(0, 134), random.randint(0, 170), random.randint(0, 134)
            patch = data[x:x+47, y:y+47, z:z+47]
            return np.expand_dims(patch, axis=0), label
        elif self.stage == 'valid':
            array_list = []
            patch_locs = [[25, 90, 30], [115, 90, 30], [67, 90, 90], [67, 45, 60], [67, 135, 60]]
            for i, loc in enumerate(patch_locs):
                x, y, z = loc
                patch = data[x:x+47, y:y+47, z:z+47]
                array_list.append(np.expand_dims(patch, axis = 0))
            data = Variable(torch.FloatTensor(np.stack(array_list, axis = 0)))
            label = Variable(torch.LongTensor([label]*5))
            return data, label
        else:
            return np.expand_dims(padding(data), axis=0), label

class dataset_longi_NACC(dataset_longi_ADNI):
    def __init__(self, Data_dir, stage):
        self.Data_dir = Data_dir
        self.stage = stage
        self.Data_list = load_txt(Data_dir, 'NACC_Longi_Data.txt')
        self.Label_list = load_txt(Data_dir, 'NACC_Longi_Label.txt')
        self.conv_index = [i for i in range(len(self.Label_list)) if self.Label_list[i]=='1']
        self.non_index = [i for i in range(len(self.Label_list)) if self.Label_list[i]=='0']
        if self.stage == 'train':
            self.con_index = self.conv_index[:70]
            self.non_index = self.non_index[:130]
        elif self.stage == 'valid':
            self.con_index = self.conv_index[70:90]
            self.non_index = self.non_index[130:171]
        elif self.stage == 'test':
            self.con_index = self.conv_index[90:]
            self.non_index = self.non_index[171:]
        elif self.stage == 'inference':
            self.con_index = self.conv_index
        self.index_list = self.con_index + self.non_index
        shuffle(self.index_list)

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        index = self.index_list[idx]
        data = np.load(self.Data_dir + self.Data_list[index])
        label = 0 if self.Label_list[index]=='0' else 1
        if self.stage == 'train':
            x, y, z = random.randint(0, 134), random.randint(0, 170), random.randint(0, 134)
            patch = data[x:x+47, y:y+47, z:z+47]
            return np.expand_dims(patch, axis=0), label
        elif self.stage == 'valid':
            array_list = []
            patch_locs = [[25, 90, 30], [115, 90, 30], [67, 90, 90], [67, 45, 60], [67, 135, 60]]
            for i, loc in enumerate(patch_locs):
                x, y, z = loc
                patch = data[x:x+47, y:y+47, z:z+47]
                array_list.append(np.expand_dims(patch, axis = 0))
            data = Variable(torch.FloatTensor(np.stack(array_list, axis = 0)))
            label = Variable(torch.LongTensor([label]*5))
            return data, label
        else:
            return np.expand_dims(padding(data), axis=0), label


