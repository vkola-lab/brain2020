#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 13:38:45 2019

@author: cxue2
"""

from model_bag import ModelBag
from utils_stat import get_roc_info, get_pr_info
import matplotlib.pyplot as plt
from utils_plot import plot_curve, plot_legend
from time import time
import numpy as np

data_root = '../tmp'
device = 'cuda:0'
K = 1  # number of models per bag
load = False

# prepare input data
print('Preparing data for models... ', end='')
Xb, X_, y = {}, {}, {}
Xb['A'], X_['A'] = {}, {}

Xb['A']['NACC_TRAIN'] = None

tmp_1 = np.load('{}/{}_riskmap.npy'.format(data_root, 'train'))
tmp_1 = tmp_1.reshape((tmp_1.shape[0], -1))
tmp_2 = np.load('{}/{}_riskmap.npy'.format(data_root, 'valid'))
tmp_2 = tmp_2.reshape((tmp_2.shape[0], -1))
X_['A']['NACC_TRAIN'] = np.concatenate((tmp_1, tmp_2))

tmp_1 = np.load('{}/{}_label.npy'.format(data_root, 'train'))
tmp_2 = np.load('{}/{}_label.npy'.format(data_root, 'valid'))
y['NACC_TRAIN'] = np.concatenate((tmp_1, tmp_2))
y['NACC_TRAIN'] = y['NACC_TRAIN'].astype(np.int)

Xb['A']['NACC_TEST'] = None

X_['A']['NACC_TEST'] = np.load('{}/{}_riskmap.npy'.format(data_root, 'test'))
X_['A']['NACC_TEST'] = X_['A']['NACC_TEST'].reshape((X_['A']['NACC_TEST'].shape[0], -1))

y['NACC_TEST'] = np.load('{}/{}_label.npy'.format(data_root, 'test'))
y['NACC_TEST'] = y['NACC_TEST'].astype(np.int)
print('Done.')

# create a bag of models
models_bag = {}
for m in ['A']:
    dimb = Xb[m]['NACC_TRAIN'].shape[1] if Xb[m]['NACC_TRAIN'] is not None else 0
    dim_ = X_[m]['NACC_TRAIN'].shape[1] if X_[m]['NACC_TRAIN'] is not None else 0 
    model_kwargs = {'dim_bn': dimb, 'dim_no_bn': dim_, 'device': device, 'balance': 1.0, 'hidden_width': 128, 'learning_rate': .01, 'batch_size': 32}
    models_bag[m] = ModelBag(model_kwargs, n_model=K)
    
# fit all models in the bag
# or load
if not load:
    for m in ['A']:
        print('Training model {}...'.format(m))
        fit_kwargs = {'X_bn': Xb[m]['NACC_TRAIN'],
                      'X_no_bn': X_[m]['NACC_TRAIN'],
                      'y': y['NACC_TRAIN'], 
                      'n_epoch': 100}
        models_bag[m].fit(fit_kwargs)
        
    # save models
    timestmp = int(time())
    for m in ['A']:
        print('Saving models... ', end='')
        models_bag[m].save('./saved_mlp/main_{}_{}'.format(timestmp, m))
        print('Done.')
else:
    print('Loading models... ', end='')
    models_bag['A'].load('./saved_mlp/main_1562094663_A')
    print('Done.')
    
# evaluate performace
scores_bag = {}
for m in ['A']:
    scores_bag[m] = {}
    for ds in ['NACC_TRAIN']:
        eval_kwargs = {'X_bn': Xb[m][ds], 'X_no_bn': X_[m][ds]}
        scores_bag[m][ds] = models_bag[m].eval(eval_kwargs)

# collect essentials for plot
roc_info, pr_info = {}, {}
for m in ['A']:
    roc_info[m], pr_info[m] = {}, {}
    for ds in ['NACC_TRAIN']:
        roc_info[m][ds] = get_roc_info(y[ds], scores_bag[m][ds][:,:,1])
        pr_info[m][ds] = get_pr_info(y[ds], scores_bag[m][ds][:,:,1])

# plot
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['figure.facecolor'] = 'w'
plt.rcParams['savefig.facecolor'] = 'w'

# roc plot
fig, axes_ = plt.subplots(figsize=[6, 6], dpi=100)
axes = dict(zip(['NACC_TRAIN'], [axes_]))
hdl_crv = {'A':{}}
for i, ds in enumerate(['NACC_TRAIN']):
    title = 'NACC (TEST)' if ds == 'NACC_TEST' else ds
    hdl_crv['A'][ds] = plot_curve(curve='roc', **roc_info['A'][ds], ax=axes[ds], **{'color':'C{}'.format(i), 'hatch':None, 'alpha':.2, 'line':'-', 'title':title})
plot_legend(axes=axes, crv_lgd_hdl=hdl_crv, crv_info=roc_info, neo_lgd_hdl={})
  
# pr plot
fig, axes_ = plt.subplots(figsize=[6, 6], dpi=100)
axes = dict(zip(['NACC_TRAIN'], [axes_]))
hdl_crv = {'A':{}}
for i, ds in enumerate(['NACC_TRAIN']):
    title = 'NACC (TEST)' if ds == 'NACC_TEST' else ds
    hdl_crv['A'][ds] = plot_curve(curve='pr', **pr_info['A'][ds], ax=axes[ds], **{'color':'C{}'.format(i), 'hatch':None, 'alpha':.2, 'line':'-', 'title':title})
plot_legend(axes=axes, crv_lgd_hdl=hdl_crv, crv_info=pr_info, neo_lgd_hdl={})