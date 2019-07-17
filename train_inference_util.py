#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 12:51:57 2019
@author: Shangranq
"""

import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
from random import shuffle
from util import *
from random import shuffle
from glob import glob
from tqdm import tqdm

def train_model_epoch(model, dataloader, optimizer, criterion):
    model.train(True)
    for inputs, labels in dataloader:
        inputs, labels = inputs.cuda(), labels.cuda()
        model.zero_grad()
        preds = model(inputs)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

def valid_model_epoch_FCN(model, dataloader):
    with torch.no_grad():
        model.train(False)
        valid_matrix = [[0, 0], [0, 0]]
        for l in range(len(dataloader)):
            inputs, labels = dataloader[l]
            inputs, labels = inputs.cuda(), labels.cuda()
            preds = model(inputs)
            valid_matrix = matrix_sum(valid_matrix, get_confusion_matrix(preds, labels))
    return valid_matrix

def valid_model_epoch_CNN(model, dataloader):
    model.train(False)
    with torch.no_grad():
        valid_matrix = [[0, 0], [0, 0]]
        for inputs, labels in dataloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            preds = model(inputs)
            valid_matrix = matrix_sum(valid_matrix, get_confusion_matrix(preds, labels))
    return valid_matrix

def train(epoches, model, dataloader_train, dataloader_valid, optimizer, criterion, model_dir):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    Optimal_valid_matrix = [[0,0],[0,0]]
    valid_accu, Epoch = 0, -1
    for epoch in range(epoches):
        train_model_epoch(model, dataloader_train, optimizer, criterion)
        valid_matrix = valid_model_epoch_FCN(model, dataloader_valid)
        print('This epoch validation confusion matrix:', valid_matrix, 'validation_accuracy:', "%.4f" % get_accu(valid_matrix))
        if get_MCC(valid_matrix) >= valid_accu:
            Epoch = epoch
            Optimal_valid_matrix = valid_matrix
            valid_accu = get_MCC(valid_matrix)
            for root, Dir, Files in os.walk(model_dir):
                for File in Files:
                    if File.endswith('.pth'):
                        os.remove(model_dir + File)
            torch.save(model.state_dict(), '{}FCN_{}.pth'.format(model_dir, Epoch))
        print('Best validation accuracy saved at the {}th epoch:'.format(Epoch), valid_accu, Optimal_valid_matrix)
    return Epoch

def inference_FCN_longi(model, model_dir, model_state_dict, dataloader, dsetname, outdir):
    model.load_state_dict(torch.load(model_dir+model_state_dict))
    fcn = model.dense_to_conv()
    Data_dir = "/data/datasets/ADNI/"
    Data_list = load_txt(Data_dir, 'ADNI_Data.txt')
    Label_list = load_txt(Data_dir, 'ADNI_Label.txt')
    NL_index_list = [i for i in range(417) if Label_list[i]=='NL']
    with torch.no_grad():
        fcn.train(False)
        for index, (input, _) in enumerate(dataloader):
            preds = fcn(input.cuda(), stage='inference')
            AD_risk = get_AD_risk(preds)
            np.save(outdir+'Risk_{}_{}.npy'.format(dsetname, NL_index_list[index]), AD_risk)

def inference_FCN(model, model_dir, model_state_dict, dataloader, dsetname, outdir):
    if not os.path.exists('./Risk_maps/'):
        os.mkdir('./Risk_maps/')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    model.load_state_dict(torch.load(model_dir + model_state_dict))
    fcn = model.dense_to_conv()
    with torch.no_grad():
        fcn.train(False)
        for index, (input, _) in tqdm(enumerate(dataloader)):
            preds = fcn(input.cuda(), stage='inference')
            AD_risk = get_AD_risk(preds)
            np.save(outdir+'Risk_{}_{}.npy'.format(dsetname, index), AD_risk)

def inference_CNN(model, model_dir, model_state_dict, dataloader, dsetname, outdir):
    model.load_state_dict(torch.load(model_dir+model_state_dict))
    features = np.zeros((len(dataloader), 64*36*8))
    with torch.no_grad():
        model.train(False)
        for index, (input, _) in enumerate(dataloader):
            preds = model(input.cuda()).data.cpu().numpy()[0, :]
            features[index] = preds
            print(dsetname, index)
    np.save(outdir+'{}_features.npy'.format(dsetname), features)

def Heatmap(risk_dir, dset_dir, stage):
    if not os.path.exists('./Heatmap/'):
        os.mkdir('./Heatmap/')
    label_list = []
    dset_name = dset_dir.split('/')[-2]
    with open(dset_dir + '{}_Label.txt'.format(dset_name), 'r') as f:
        for line in f:
            label_list.append(line.strip('\n'))
    if dset_name == 'ADNI' and stage == 'train':
        risk_list = ['Risk_{}_{}.npy'.format(dset_name, i) for i in range(257)]
        label_list = label_list[:257]
    if dset_name == 'ADNI' and stage == 'valid':
        risk_list = ['Risk_{}_{}.npy'.format(dset_name, i) for i in range(257, 337)]
        label_list = label_list[257:337]
    if dset_name == 'ADNI' and stage == 'test':
        risk_list = ['Risk_{}_{}.npy'.format(dset_name, i) for i in range(337, 417)]
        label_list = label_list[337:]
    if dset_name != 'ADNI':
        risk_list = ['Risk_{}_{}.npy'.format(dset_name, i) for i in range(len(label_list))]

    TP, FP, TN, FN = np.zeros((46, 55, 46)), np.zeros((46, 55, 46)), np.zeros((46, 55, 46)), np.zeros((46, 55, 46))
    for label, risk in zip(label_list, risk_list):
        risk_map = np.load(risk_dir + risk)
        if label == 'NL':
            TN += (risk_map<0.5).astype(np.int)
            FP += (risk_map>=0.5).astype(np.int)
        elif label == 'AD':
            TP += (risk_map>=0.5).astype(np.int)
            FN += (risk_map<0.5).astype(np.int)
    count = len(label_list)
    TP, TN, FP, FN = TP.astype(np.float)/count, TN.astype(np.float)/count, FP.astype(np.float)/count, FN.astype(np.float)/count
    ACCU = TP + TN
    F1 = 2*TP / (2*TP+FP+FN)
    MCC = (TP*TN - FP*FN) / (np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))+0.00000001*np.ones((46, 55, 46)))
    np.save('./Heatmap/{}_{}_MCC.npy'.format(dset_name, stage), MCC)
    np.save('./Heatmap/{}_{}_F1.npy'.format(dset_name, stage), F1)
    np.save('./Heatmap/{}_{}_ACCU.npy'.format(dset_name, stage), ACCU) 

def Heatmap_Longi(risk_dir, Data_dir, stage):
    dset_name = Data_dir.split('/')[-2]
    Data_list = load_txt(Data_dir, 'NACC_Longi_Data.txt')
    Label_list = load_txt(Data_dir, 'NACC_Longi_Label.txt')
    conv_index = [i for i in range(len(Label_list)) if Label_list[i]=='1']
    non_index = [i for i in range(len(Label_list)) if Label_list[i]=='0']
    if stage == 'train':
         con_index = conv_index[:70]
         non_index = non_index[:130]
    elif stage == 'valid':
         con_index = conv_index[70:90]
         non_index = non_index[130:171]
    elif stage == 'test':
         con_index = conv_index[90:]
         non_index = non_index[171:]
    index = con_index + non_index
    risk_list = ['Risk_{}_{}.npy'.format(dset_name, i) for i in index]
    label_list = [Label_list[i] for i in index]

    TP, FP, TN, FN = np.zeros((46, 55, 46)), np.zeros((46, 55, 46)), np.zeros((46, 55, 46)), np.zeros((46, 55, 46))
    for label, risk in zip(label_list, risk_list):
        risk_map = np.load(risk_dir + risk)
        if label == '0':
            TN += (risk_map<0.5).astype(np.int)
            FP += (risk_map>=0.5).astype(np.int)
        elif label == '1':
            TP += (risk_map>=0.5).astype(np.int)
            FN += (risk_map<0.5).astype(np.int)
    count = len(label_list)
    TP, TN, FP, FN = TP.astype(np.float)/count, TN.astype(np.float)/count, FP.astype(np.float)/count, FN.astype(np.float)/count
    ACCU = TP + TN
    F1 = 2*TP / (2*TP+FP+FN)
    MCC = (TP*TN - FP*FN) / (np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))+0.00000001*np.ones((46, 55, 46)))
    np.save('./Heatmap/{}_{}_MCC.npy'.format(dset_name, stage), MCC)
    np.save('./Heatmap/{}_{}_F1.npy'.format(dset_name, stage), F1)
    np.save('./Heatmap/{}_{}_ACCU.npy'.format(dset_name, stage), ACCU)

        
























