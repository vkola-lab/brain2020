from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from utils import PatchGenerator, padding, read_csv, get_AD_risk
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
    def __init__(self, X_bn, X_no_bn, y):
        self.X_bn = X_bn
        self.X_no_bn = X_no_bn
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        X_bn_sample = self.X_bn[idx] if self.X_bn is not None else None
        X_no_bn_sample = self.X_no_bn[idx] if self.X_no_bn is not None else None
        y_sample = self.y[idx]
        return X_bn_sample, X_no_bn_sample, y_sample
    
def _MLP_collate_fn(batch):
    X_bn_batch = np.stack([s[0] for s in batch]) if batch[0][0] is not None else None
    X_no_bn_batch = np.stack([s[1] for s in batch]) if batch[0][1] is not None else None
    y_batch = np.stack([s[2] for s in batch])
    return X_bn_batch, X_no_bn_batch, y_batch

class BuildDF:
    def __init__(self, exp_idx, roi_thrshold=0.6):
        self.roi_thrshold = roi_thrshold
        self.exp_idx = exp_idx
        self.select_roi()
        
    def select_roi(self):
        self.roi = np.load('./DPMs/fcn_exp{}/train_MCC.npy'.format(self.exp_idx))
        self.roi = self.roi > self.roi_thrshold
        for i in range(self.roi.shape[0]):
            for j in range(self.roi.shape[1]):
                for k in range(self.roi.shape[2]):
                    if i%3!=0 or j%2!=0 or k%3!=0:
                        self.roi[i,j,k] = False
        
    def load(self):
        # read csv into data frame
        tmp = '/home/sq/brain2020'
        table = []
        for stage in ['train', 'valid', 'test']:
            with open('{}/lookupcsv/exp{}/{}.csv'.format(tmp, self.exp_idx, stage), newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    #risk = get_AD_risk(np.load('{}/DPMs/fcn_exp{}/'.format(tmp, self.exp_idx) + row['filename'] + '.npy'))
                    risk = None
                    line = list(row.values()) + [stage] + [risk]
                    table.append(line)
        for stage in ['NACC', 'AIBL']:
            with open('{}/lookupcsv/{}.csv'.format(tmp, stage), newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    #risk = get_AD_risk(np.load('{}/DPMs/fcn_exp{}/'.format(tmp, self.exp_idx) + row['filename'] + '.npy'))
                    risk = None
                    line = list(row.values()) + [stage] + [risk]
                    table.append(line)
        self.df = pd.DataFrame(table, columns = ['filename', 'label', 'age', 'gender', 'mmse', 'apoe', 'ravlt_lrn', 'ravlt_fgt', 'ravlt_pfgt', 'dataset', 'riskmap'])
            
        # AD, NL to 1, 0
        label_dict = {'AD':1, 'NL':0}
        for i in range(len(self.df)):
            self.df.loc[i, 'label'] = label_dict[self.df.loc[i, 'label']]
    
    def get_ndarray(self, cols, longitudinal=False):
        dset_names = self.df.dataset.unique().tolist()
        cols_bn = sorted(list(set(cols) & {'age', 'mmse', 'ravlt_lrn', 'ravlt_fgt', 'ravlt_pfgt'}))
        cols_no_bn = sorted(list(set(cols) & {'riskmap', 'gender', 'apoe'}))
        X_bn, X_no_bn, y = {}, {}, {}
        for ds in dset_names:
            # X that needs to be batch-normalized
            if cols_bn:
                X_bn[ds] = self.df[self.df.dataset == ds][cols_bn].values
                X_bn[ds] = [np.concatenate(r) for r in X_bn[ds]]
                X_bn[ds] = np.stack(X_bn[ds]).astype(np.float32)
            else: 
                X_bn[ds] = None
                
            # X that doesn't need to be batch-normalized
            if cols_no_bn:
                X_no_bn[ds] = self.df[self.df.dataset == ds][cols_no_bn].values
                X_no_bn[ds] = [np.concatenate(r) for r in X_no_bn[ds]]
                X_no_bn[ds] = np.stack(X_no_bn[ds]).astype(np.float32)
            else: 
                X_no_bn[ds] = None

            # y
            if not longitudinal:
                y[ds] = self.df[self.df.dataset == ds]['label'].values.astype(np.int)
            else:
                y[ds] = self.df[self.df.dataset == ds]['long_label'].values.astype(np.int)
            
        return X_bn, X_no_bn, y
    
    def keep_data_completeness(self, cols):
        count_pre = self.df['dataset'].value_counts()
        len_old = len(self.df)
        mask = pd.notnull(self.df[cols]).all(axis=1)
        self.df = self.df[mask]
        count_new = self.df['dataset'].value_counts()
        print('Number of deleted rows: {}/{}'.format(len_old-len(self.df), len_old))
        for k, v in count_pre.items():
            if k in count_new:
                print('\t{}: {}/{}'.format(k, v - self.df['dataset'].value_counts()[k], v))
            else:
                print('\t{}: {}/{}'.format(k, v, v))


def load_neurologist_data(fn):
    with open(fn, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        rows = list(csv_reader)
    rows = np.array(rows)
    rows = rows[1:,4:]
    rows[rows == 'AD'] = 1
    rows[rows == 'NL'] = 0
    
    rslt = {'y': rows[:,0].astype(np.int),
            'y_pred_list': rows[:,1:].T.astype(np.int)}
    return rslt


if __name__ == "__main__":
    dw = BuildDF(exp_idx=0)
    dw.load()
    print(dw.df)
#    dataset = CNN_Data(Data_dir='/data/datasets/ADNI_NoBack/', stage='train')
#    for i in range(len(dataset)):
#        scan, label = dataset[i]
#        print(scan.shape, label.shape)
