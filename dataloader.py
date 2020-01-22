from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from utils import PatchGenerator, padding, read_csv, read_csv_complete, get_AD_risk
import random
import pandas as pd
import csv

"""
dataloaders are defined in this scripts:

    1. FCN dataloader (data split into 60% train, 20% validation and 20% testing)
        (a). Training stage:    use random patches to train classification FCN model 
        (b). Validation stage:  forward whole volume MRI to FCN to get Disease Probability Map (DPM). use MCC of DPM as criterion to save model parameters   
        (c). Testing stage:     get all available DPMs for the development of MLP 
    
    2. MLP dataloader (use the exactly same split as FCN dataloader)
        (a). Training stage:    train MLP on DPMs from the training portion
        (b). Validation stage:  use MCC as criterion to save model parameters   
        (c). Testing stage:     test the model on ADNI_test, NACC, FHS and AIBL datasets
        
    3. CNN dataloader (baseline classification model to be compared with FCN+MLP framework)
        (a). Training stage:    use whole volume to train classification FCN model 
        (b). Validation stage:  use MCC as criterion to save model parameters   
        (c). Testing stage:     test the model on ADNI_test, NACC, FHS and AIBL datasets
"""


class CNN_Data(Dataset):
    """
    csv files ./lookuptxt/*.csv contains MRI filenames along with demographic and diagnosis information 
    MRI with clip and backremove: /data/datasets/ADNI_NoBack/*.npy
    """
    def __init__(self, Data_dir, exp_idx, stage, seed=1000):
        random.seed(seed)
        self.Data_dir = Data_dir
        if stage in ['train', 'valid', 'test']:
            self.Data_list, self.Label_list = read_csv('./lookupcsv/exp{}/{}.csv'.format(exp_idx, stage))
        elif stage in ['ADNI', 'NACC', 'AIBL', 'FHS']:
            self.Data_list, self.Label_list = read_csv('./lookupcsv/{}.csv'.format(stage))

    def __len__(self):
        return len(self.Data_list)

    def __getitem__(self, idx):
        label = self.Label_list[idx]
        data = np.load(self.Data_dir + self.Data_list[idx] + '.npy').astype(np.float32)
        data = np.expand_dims(data, axis=0)
        return data, label

    def get_sample_weights(self):
        count, count0, count1 = float(len(self.Label_list)), float(self.Label_list.count(0)), float(self.Label_list.count(1))
        weights = [count / count0 if i == 0 else count / count1 for i in self.Label_list]
        return weights, count0 / count1


class FCN_Data(CNN_Data):
    def __init__(self, Data_dir, exp_idx, stage, whole_volume=False, seed=1000, patch_size=47):
        CNN_Data.__init__(self, Data_dir, exp_idx, stage, seed)
        self.stage = stage
        self.whole = whole_volume
        self.patch_size = patch_size
        self.patch_sampler = PatchGenerator(patch_size=self.patch_size)

    def __getitem__(self, idx):
        label = self.Label_list[idx]
        data = np.load(self.Data_dir + self.Data_list[idx] + '.npy').astype(np.float32)
        if self.stage == 'train' and not self.whole:
            patch = self.patch_sampler.random_sample(data)
            patch = np.expand_dims(patch, axis=0)
            return patch, label
        else:
            data = np.expand_dims(padding(data, win_size=self.patch_size // 2), axis=0)
            return data, label


class MLP_Data(Dataset):
    def __init__(self, Data_dir, exp_idx, stage, roi_threshold, roi_count, choice, seed=1000):
        random.seed(seed)
        self.exp_idx = exp_idx
        self.Data_dir = Data_dir
        self.roi_threshold = roi_threshold
        self.roi_count = roi_count
        if choice == 'count':
            self.select_roi_count()
        else:
            self.select_roi_thres()
        if stage in ['train', 'valid', 'test']:
            path = './lookupcsv/exp{}/{}.csv'.format(exp_idx, stage)
        else:
            path = './lookupcsv/{}.csv'.format(stage)
        self.Data_list, self.Label_list, self.demor_list = read_csv_complete(path)
        self.risk_list = [get_AD_risk(np.load(Data_dir+filename+'.npy'))[self.roi] for filename in self.Data_list]
        self.in_size = self.risk_list[0].shape[0]
        
    def select_roi_thres(self):
        self.roi = np.load('./DPMs/fcn_exp{}/train_MCC.npy'.format(self.exp_idx))
        self.roi = self.roi > self.roi_threshold
        for i in range(self.roi.shape[0]):
            for j in range(self.roi.shape[1]):
                for k in range(self.roi.shape[2]):
                    if i%3!=0 or j%2!=0 or k%3!=0:
                        self.roi[i,j,k] = False

    def select_roi_count(self):
        self.roi = np.load('./DPMs/fcn_exp{}/train_MCC.npy'.format(self.exp_idx))
        tmp = []
        for i in range(self.roi.shape[0]):
            for j in range(self.roi.shape[1]):
                for k in range(self.roi.shape[2]):
                    if i%3!=0 or j%2!=0 or k%3!=0: continue
                    tmp.append((self.roi[i,j,k], i, j, k))
        tmp.sort()
        tmp = tmp[-self.roi_count:]
        self.roi = self.roi != self.roi
        for _, i, j, k in tmp:
            self.roi[i,j,k] = True

    def __len__(self):
        return len(self.Data_list)

    def __getitem__(self, idx):
        label = self.Label_list[idx]
        risk = self.risk_list[idx]
        demor = self.demor_list[idx]
        return risk, label, np.asarray(demor).astype(np.float32)

    def get_sample_weights(self):
        count, count0, count1 = float(len(self.Label_list)), float(self.Label_list.count(0)), float(self.Label_list.count(1))
        weights = [count / count0 if i == 0 else count / count1 for i in self.Label_list]
        return weights, count0 / count1


if __name__ == "__main__":
    data = MLP_Data(Data_dir='./DPMs/fcn_exp1/', exp_idx=1, stage='train', roi_threshold=0.6)
    dataloader = DataLoader(data, batch_size=10, shuffle=False)
    for risk, label, demor in dataloader:
        print(risk.shape, label, demor)