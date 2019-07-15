# -*- coding: utf-8 -*-
"""
Created on Mon May 27 12:52:23 2019

@author: Iluva
"""

import pandas as pd
import numpy as np
import csv

class AlzheimerDataWrapper():
    def __init__(self):
        self.df = pd.DataFrame(columns = ['dataset', 'X_rmap', 'X_age', 'X_mmse', 'X_gender', 'label', 'point', 'id', 'X_apoe', 'X_ravlt_lrn', 'X_ravlt_fgt', 'X_ravlt_pfgt', 'X_cnn'])
        
    def load(self, root, dsets, roi, fn_3T=None, longitudinal=False):
        label_dict = {'AD':1, 'NL':0}
        ADNI_cut = {'train':257, 'valid':337}
        
        for ds in dsets:
            with open('../{}_Risk/{}_Label.txt'.format(ds, ds), 'r') as f:
                labels = [l.strip('\n') for l in f]
            with open('../{}_Risk/{}_Age.txt'.format(ds, ds), 'r') as f:
                ages = [[float(l.strip('\n'))] for l in f]
            with open('../{}_Risk/{}_MMSE.txt'.format(ds, ds), 'r') as f:
                mmse = [[float(l.strip('\n'))] if l.strip('\n') != '' else None for l in f]
            with open('../{}_Risk/{}_GENDER.txt'.format(ds, ds), 'r') as f:
                gender = [np.array([1.,0.]) if int(l.strip('\n')) == 1 else np.array([0.,1.]) for l in f]
            with open('../{}_Risk/{}_points.txt'.format(ds, ds), 'r') as f:
                point = [0 if l.strip('\n')=='a' else int(l.strip('\n')) for l in f]       
            with open('../{}_Risk/{}_Apoe.txt'.format(ds, ds), 'r') as f:
                apoe = [[float(l.strip('\n'))] if l.strip('\n') != '' else None for l in f]
            cnn = np.load('../{}_Risk/{}_features.npy'.format(ds, ds))
            if ds == 'ADNI':
                with open('../{}_Risk/{}_RAVLT.txt'.format(ds, ds), 'r') as f:
                    ravlt = []
                    for l in f:
                        if l.strip('\n') == '':
                            ravlt.append(None)
                        else:
                            tmp = l.strip('\n').split('_')
                            ravlt.append([[float(e)] for e in tmp])
            else:
                ravlt = None
            
            # some cases need to be excluded
            if ds == 'NACC':
                with open('../NACC_Risk/NACC_Check_MLP.txt') as f:
                    nacc_check = [True if l.strip('\n') == 'True' else False for l in f]
                    
            if ds == 'FHS':
                with open('../FHS_Risk/FHS_Check_MLP.txt') as f:
                    fhs_check = [True if l.strip('\n') == 'True' else False for l in f]
            
            # append to Pandas dataframe
            row = [None] * 13
            for i in range(len(labels)):
                if ds == 'ADNI':
                    if i < ADNI_cut['train']:
                        row[0] = ds + '_TRAIN'
                        fn_str = '../{}_Risk/riskTrain{}_{}.npy'
                        row[1] = np.load(fn_str.format(ds, i, labels[i]))[roi]
                    elif i < ADNI_cut['valid']:
                        row[0] = ds + '_TRAIN'
                        fn_str = '../{}_Risk/riskValid{}_{}.npy'
                        row[1] = np.load(fn_str.format(ds, i-ADNI_cut['train'], labels[i]))[roi]
                    else:
                        row[0] = ds + '_TEST'
                        fn_str = '../{}_Risk/riskTest{}_{}.npy'
                        row[1] = np.load(fn_str.format(ds, i-ADNI_cut['valid'], labels[i]))[roi]
                else:
                    if ds == 'NACC' and not nacc_check[i]:
                        continue
                    
                    if ds == 'FHS' and not fhs_check[i]:
                        continue
                    
                    row[0] = ds
                    fn_str = '../{}_Risk/risk{}{}_{}.npy'
                    row[1] = np.load(fn_str.format(ds, ds, i, labels[i]))[roi]
                row[2] = ages[i]
                row[3] = mmse[i]
                row[4] = gender[i]
                row[5] = labels[i]
                row[6] = point[i]
                row[7] = i
                row[8] = apoe[i]
                if ravlt is not None:
                    row[9] = ravlt[i][0] if ravlt[i] is not None else None
                    row[10] = ravlt[i][1] if ravlt[i] is not None else None
                    row[11] = ravlt[i][2] if ravlt[i] is not None else None
                else:
                    row[9] = None
                    row[10] = None
                    row[11] = None
                row[12] = cnn[i,:]
                self.df.loc[len(self.df)] = row
            
        # AD, NL to 0, 1
        for i in range(len(self.df)):
            self.df.loc[i, 'label'] = label_dict[self.df.loc[i, 'label']]
            
        if fn_3T:
            self._load_3T(fn_3T, roi)
            
        if longitudinal:
            self._load_longitudinal()
            
    def _load_3T(self, fn, roi):
        with open(fn) as csv_file:
            csv_reader = csv.reader(csv_file)
            for i, r in enumerate(csv_reader):
                if i == 0:
                    continue
                pid = r[1]
                rmap_1T = np.load('../ADNI_overlap/Risk_ADNI_{}.npy'.format(pid))[roi]
                rmap_3T = np.load('../ADNI_3T_overlap/Risk_ADNI_3T_{}.npy'.format(pid))[roi]
                age = [float(r[5])]
                label = 1 if r[2] == 'AD' else 0
                mmse = [float(r[6])]
                gender = [1., 0.] if r[7] == '2' else [0., 1.]
                
                row_1T = ['ADNI_1.5T', rmap_1T, age, mmse, gender, label,
                          None, None, None, None, None, None, None]
                row_3T = ['ADNI_3.0T', rmap_3T, age, mmse, gender, label,
                          None, None, None, None, None, None, None]
                self.df.loc[len(self.df)] = row_1T
                self.df.loc[len(self.df)] = row_3T
    
    def _load_longitudinal(self):
        self.df = self.df[(self.df.dataset == 'ADNI_TEST') | (self.df.dataset == 'ADNI_TRAIN')]
        with open('../ADNI_Risk/ADNI_Longi_label.txt') as f:
            labels = [l.strip('\n') for l in f]
        labels = [int(e) if e in ['0', '1'] else -1 for e in labels]
        self.df['long_label'] = labels
        self.df = self.df[self.df.long_label != -1]
    
    def get_ndarray(self, cols, longitudinal=False):
        dset_names = self.df.dataset.unique().tolist()
        cols_bn = sorted(list(set(cols) & {'X_age', 'X_mmse', 'X_ravlt_lrn', 'X_ravlt_fgt', 'X_ravlt_pfgt'}))
        cols_no_bn = sorted(list(set(cols) & {'X_rmap', 'X_gender', 'X_apoe', 'X_cnn'}))
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


class LongitudinalDataWrapper():
    def __init__(self):
        self.df = pd.DataFrame(columns = ['dataset', 'X_rmap', 'X_age', 'X_mmse', 'X_gender', 'label', 'point', 'id', 'X_apoe', 'X_ravlt_lrn', 'X_ravlt_fgt', 'X_ravlt_pfgt', 'X_cnn'])
    
            
def select_roi(roi_src_fn, roi_thrshold=0.6):
    roi = np.load(roi_src_fn)
    roi = roi > roi_thrshold
    for i in range(roi.shape[0]):
        for j in range(roi.shape[1]):
            for k in range(roi.shape[2]):
                if i%3!=0 or j%2!=0 or k%3!=0:
                    roi[i,j,k] = False
    return roi

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

