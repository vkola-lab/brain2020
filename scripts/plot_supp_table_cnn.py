# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 01:43:06 2019

@author: Iluva
"""

from utils_data import AlzheimerDataWrapper, select_roi
from model_bag import ModelBag
from utils_stat import calc_performance_statistics
import matplotlib.pyplot as plt
from time import time
import numpy as np

data_root = '../'
device = 'cpu'
roi_src_fn = '../metric/TRAIN_MCC.npy'
roi_thrshold = 0.6
dsets = ['ADNI', 'FHS', 'AIBL', 'NACC']
K = 100  # number of models per bag
load = True

# ROI
print('Constructing region of importance... ', end='')
roi = select_roi(roi_src_fn, roi_thrshold=0.6)
print('Done.')

# load data into dataframe
print('Loading data into Pandas dataframe... ', end='')
dw = AlzheimerDataWrapper()
dw.load(data_root, dsets, roi)
print('Done.')

# remove rows with missing values
dw.keep_data_completeness(['X_mmse'])

# prepare input data
# select columns from dataframe to form ndarray
print('Preparing data for models ... ', end='')
Xb, X_, y = {}, {}, {}
for m, c in zip(['A', 'C'],
                [['X_cnn'],
                 ['X_cnn', 'X_mmse', 'X_age', 'X_gender']]):
    Xb[m], X_[m], y = dw.get_ndarray(cols=c)
print('Done.')

# create a bag of models
models_bag = {}
for m in ['A', 'C']:
    dimb = Xb[m]['ADNI_TRAIN'].shape[1] if Xb[m]['ADNI_TRAIN'] is not None else 0
    dim_ = X_[m]['ADNI_TRAIN'].shape[1] if X_[m]['ADNI_TRAIN'] is not None else 0 
    model_kwargs = {'dim_bn': dimb, 'dim_no_bn': dim_, 'device': device}
    models_bag[m] = ModelBag(model_kwargs, n_model=K)

# fit all models in the bag
# or load
if not load:
    for m in ['A', 'C']:
        print('Training model {}...'.format(m))
        fit_kwargs = {'X_bn': Xb[m]['ADNI_TRAIN'],
                      'X_no_bn': X_[m]['ADNI_TRAIN'],
                      'y': y['ADNI_TRAIN'], 
                      'n_epoch': 200}
        models_bag[m].fit(fit_kwargs)
        
    # save models
    timestmp = int(time())
    for m in ['A', 'C']:
        print('Saving models... ', end='')
        models_bag[m].save('./saved_mlp/main_{}_{}'.format(timestmp, m))
        print('Done.')
else:
    print('Loading models... ', end='')
    models_bag['A'].load('./saved_mlp/main_1561440550_A')
    models_bag['C'].load('./saved_mlp/main_1561440550_C')
    print('Done.')

# evaluate performace
scores_bag = {}
for m in ['A', 'C']:
    scores_bag[m] = {}
    for ds in ['ADNI_TEST', 'AIBL', 'FHS', 'NACC']:
        eval_kwargs = {'X_bn': Xb[m][ds], 'X_no_bn': X_[m][ds]}
        scores_bag[m][ds] = models_bag[m].eval(eval_kwargs)
        
# get performance statistics
stat = {}
for m in ['A', 'C']:
    stat[m] = {}
    for ds in ['ADNI_TEST', 'FHS', 'AIBL', 'NACC']:
        scores = scores_bag[m][ds]
        stat[m][ds] = calc_performance_statistics(scores, y[ds])
        
# calculate the mean and std of the performance statistics
stat_avg = {}
for m in ['A', 'C']:
    stat_avg[m] = {}
    for ds in ['ADNI_TEST', 'FHS', 'AIBL', 'NACC']:
        stat_avg[m][ds] = {}
        for k in stat[m][ds]:
            if k != 'confusion_matrix':
                mean = np.mean(stat[m][ds][k])
                std = np.std(stat[m][ds][k])
                stat_avg[m][ds][k] = {'mean':mean, 'std':std}
                
# table entries
tab = {}
for m in ['A', 'C']:
    tab[m] = []
    for ds in ['ADNI_TEST', 'AIBL', 'FHS', 'NACC']:
        row = []
        for k in ['accuracy', 'sensitivity', 'specificity', 'f1_score', 'MCC']:
            row.append('{:.03f}$\pm${:.03f}'.format(stat_avg[m][ds][k]['mean'], stat_avg[m][ds][k]['std']))
        tab[m].append(row)

fig, axs_ = plt.subplots(2, 1)
axs = dict(zip(['A', 'C'], axs_))
for m in ['A', 'C']:
    axs[m].table(cellText=tab[m],
              colLabels=['Accuracy', 'Sensitivity', 'Specificity', 'F1-score', 'MCC'],
              rowLabels=['ADNI Test', 'AIBL', 'FHS', 'NACC'],
              loc='center', cellLoc='center')
    axs[m].set_title('Model {}'.format(m))
    axs[m].axis('tight')
    axs[m].axis('off')