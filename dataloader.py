from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
from utils import PatchGenerator, padding
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
    txt files ./lookuptxt/*.txt complete path of MRIs
    MRI with clip and backremove: /data/datasets/ADNI_NoBack/*.npy
    """
    def __init__(self, Data_dir, class1, class2, stage, ratio=(0.6, 0.2, 0.2), seed=1000):
        random.seed(seed)
        self.Data_dir = Data_dir
        Data_list0 = read_txt('../lookuptxt/', class1 + '.txt')
        Data_list1 = read_txt('../lookuptxt/', class2 + '.txt')
        self.Data_list = Data_list0 + Data_list1
        self.Label_list = [0]*len(Data_list0) + [1]*len(Data_list1)
        length = len(self.Data_list)
        idxs = list(range(length))
        random.shuffle(idxs)
        split1, split2 = int(length*ratio[0]), int(length*(ratio[0]+ratio[1]))
        self.stage = stage
        if self.stage == 'train':
            self.index_list = idxs[:split1]
        elif self.stage == 'valid':
            self.index_list = idxs[split1:split2]
        elif self.stage == 'test':
            self.index_list = idxs[split2:]
        else:
            raise ValueError('invalid stage setting')

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        index = self.index_list[idx]
        label = self.Label_list[index]
        data = np.load(self.Data_dir + self.Data_list[index]).astype(np.float32)
        data = np.expand_dims(data, axis=0)
        return data, label

    def get_sample_weights(self):
        labels = []
        for idx in self.index_list:
            labels.append(self.Label_list[idx])
        count, count0, count1 = float(len(labels)), float(labels.count(0)), float(labels.count(1))
        weights = [count / count0 if i == 0 else count / count1 for i in labels]
        return weights, count0 / count1


class FCN_Data(CNN_Data):

    def __init__(self, Data_dir, class1, class2, stage, ratio=(0.6, 0.2, 0.2), seed=1000, patch_size=47):
        CNN_Data.__init__(self, Data_dir, class1, class2, stage, ratio=(0.6, 0.2, 0.2), seed=1000)
        self.patch_size = patch_size
        self.patch_sampler = PatchGenerator(patch_size=self.patch_size)

    def __getitem__(self, idx):
        index = self.index_list[idx]
        label = self.Label_list[index]
        data = np.load(self.Data_dir + self.Data_list[index]).astype(np.float32)
        if self.stage == 'test' or self.stage == 'valid':
            data = np.expand_dims(padding(data, margin_width=self.patch_size//2), axis=0)
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
