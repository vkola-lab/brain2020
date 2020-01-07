from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from utils import PatchGenerator, padding, read_csv
import random
import copy

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
        self.Data_list, self.Label_list = read_csv('./lookupcsv/exp{}/{}.csv'.format(exp_idx, stage))

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
    def __init__(self, Data_dir, exp_idx, stage, seed=1000, patch_size=47):
        CNN_Data.__init__(self, Data_dir, exp_idx, stage, seed=1000)
        self.stage = stage
        self.patch_size = patch_size
        self.patch_sampler = PatchGenerator(patch_size=self.patch_size)

    def __getitem__(self, idx):
        label = self.Label_list[idx]
        data = np.load(self.Data_dir + self.Data_list[idx] + '.npy').astype(np.float32)
        if self.stage == 'test' or self.stage == 'valid':
            data = np.expand_dims(padding(data, win_size=self.patch_size//2), axis=0)
            return data, label
        elif self.stage == 'train':
            patch = self.patch_sampler.random_sample(data)
            patch = np.expand_dims(patch, axis=0)
            return patch, label


class MLP_Data(Dataset):
    def __init__(self):
        pass



if __name__ == "__main__":
    dataset = CNN_Data(Data_dir='/data/datasets/ADNI_NoBack/', class1='ADNI_1.5T_GAN_NL', class2='ADNI_1.5T_GAN_AD', stage='train')
    for i in range(len(dataset)):
        scan, label = dataset[i]
        print(scan.shape, label.shape)
