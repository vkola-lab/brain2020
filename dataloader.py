from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils import PatchGenerator, padding, read_csv, read_csv_complete, read_csv_complete_apoe, get_AD_risk
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

class Augment:
    def __init__(self):
        self.contrast_factor = 0.2
        self.bright_factor = 0.4
        self.sig_factor = 0.2

    def change_contrast(self, image):
        ratio = 1 + (random.random() - 0.5)*self.contrast_factor
        return image.mean() + ratio*(image - image.mean())

    def change_brightness(self, image):
        val = (random.random() - 0.5)*self.bright_factor
        return image + val

    def add_noise(self, image):
        sig = random.random() * self.sig_factor
        return np.random.normal(0, sig, image.shape) + image

    def apply(self, image):
        image = self.change_contrast(image)
        image = self.change_brightness(image)
        image = self.add_noise(image)
        return image


class CNN_Data(Dataset):
    """
    csv files ./lookuptxt/*.csv contains MRI filenames along with demographic and diagnosis information 
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
    def __init__(self,
                 Data_dir,
                 exp_idx,
                 stage,
                 whole_volume=False,
                 seed=1000,
                 patch_size=47,
                 transform=Augment()):

        """
        :param Data_dir:      data path
        :param exp_idx:       experiment index maps to different data splits
        :param stage:         stage could be 'train', 'valid', 'test' and etc ...
        :param whole_volume:  if whole_volume == True, get whole MRI;
                              if whole_volume == False and stage == 'train', sample patches for training
        :param seed:          random seed
        :param patch_size:    patch size has to be 47, otherwise model needs to be changed accordingly
        :param transform:     transform is about data augmentation, if transform == None: no augmentation
                              for more details, see Augment class
        """

        CNN_Data.__init__(self, Data_dir, exp_idx, stage, seed)
        self.stage = stage
        self.transform = transform
        self.whole = whole_volume
        self.patch_size = patch_size
        self.patch_sampler = PatchGenerator(patch_size=self.patch_size)

    def __getitem__(self, idx):
        label = self.Label_list[idx]
        if self.stage == 'train' and not self.whole:
            data = np.load(self.Data_dir + self.Data_list[idx] + '.npy', mmap_mode='r').astype(np.float32)
            patch = self.patch_sampler.random_sample(data)
            if self.transform:
                patch = self.transform.apply(patch).astype(np.float32)
            patch = np.expand_dims(patch, axis=0)
            return patch, label
        else:
            data = np.load(self.Data_dir + self.Data_list[idx] + '.npy').astype(np.float32)
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
            self.path = './lookupcsv/exp{}/{}.csv'.format(exp_idx, stage)
        else:
            self.path = './lookupcsv/{}.csv'.format(stage)
        self.Data_list, self.Label_list, self.demor_list = read_csv_complete(self.path)
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


class MLP_Data_apoe(MLP_Data):
    def __init__(self, Data_dir, exp_idx, stage, roi_threshold, roi_count, choice, seed=1000):
        super().__init__(Data_dir, exp_idx, stage, roi_threshold, roi_count, choice, seed)
        self.Data_list, self.Label_list, self.demor_list = read_csv_complete_apoe(self.path)


class CNN_MLP_Data(Dataset):
    def __init__(self, Data_dir, exp_idx, stage, seed=1000):
        random.seed(seed)
        self.exp_idx = exp_idx
        self.Data_dir = Data_dir
        if stage in ['train', 'valid', 'test']:
            path = './lookupcsv/exp{}/{}.csv'.format(exp_idx, stage)
        else:
            path = './lookupcsv/{}.csv'.format(stage)
        self.Data_list, self.Label_list, self.demor_list = read_csv_complete(path)
        self.risk_list = [np.load(Data_dir + filename + '.npy') for filename in self.Data_list]
        self.risk_list = [self.rescale(a) for a in self.risk_list]
        self.in_size = self.risk_list[0].shape[0]

    def __len__(self):
        return len(self.Data_list)

    def __getitem__(self, idx):
        label = self.Label_list[idx]
        risk = self.risk_list[idx]
        demor = self.demor_list[idx]
        return risk, label, np.asarray(demor).astype(np.float32)

    def rescale(self, x):
        return (x + 8) / 20.0

    def get_sample_weights(self):
        count, count0, count1 = float(len(self.Label_list)), float(self.Label_list.count(0)), float(
            self.Label_list.count(1))
        weights = [count / count0 if i == 0 else count / count1 for i in self.Label_list]
        return weights, count0 / count1


if __name__ == "__main__":
    data = CNN_MLP_Data(Data_dir='./DPMs/cnn_exp1/', exp_idx=1, stage='train')
    dataloader = DataLoader(data, batch_size=10, shuffle=False)
    for risk, label, demor in dataloader:
        print(risk.shape, label, demor)