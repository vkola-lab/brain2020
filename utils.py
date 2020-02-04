import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from numpy import random
import json
import csv
import random
import os
import time


class PatchGenerator:
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def random_sample(self, volume):
        """sample random patch from numpy array data"""
        X, Y, Z = volume.shape
        x = random.randint(0, X-self.patch_size)
        y = random.randint(0, Y-self.patch_size)
        z = random.randint(0, Z-self.patch_size)
        return volume[x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size]

    def fixed_sample(self, data):
        """sample patch from fixed locations"""
        patches = []
        patch_locs = [[25, 90, 30], [115, 90, 30], [67, 90, 90], [67, 45, 60], [67, 135, 60]]
        for i, loc in enumerate(patch_locs):
            x, y, z = loc
            patch = data[x:x+47, y:y+47, z:z+47]
            patches.append(np.expand_dims(patch, axis = 0))
        return patches


def load_txt(txt_dir, txt_name):
    List = []
    with open(txt_dir + txt_name, 'r') as f:
        for line in f:
            List.append(line.strip('\n').replace('.nii', '.npy'))
    return List


def padding(tensor, win_size=23):
    A = np.ones((tensor.shape[0]+2*win_size, tensor.shape[1]+2*win_size, tensor.shape[2]+2*win_size)) * tensor[-1,-1,-1]
    A[win_size:win_size+tensor.shape[0], win_size:win_size+tensor.shape[1], win_size:win_size+tensor.shape[2]] = tensor
    return A.astype(np.float32)


def get_confusion_matrix(preds, labels):
    labels = labels.data.cpu().numpy()
    preds = preds.data.cpu().numpy()
    matrix = [[0, 0], [0, 0]]
    for index, pred in enumerate(preds):
        if np.amax(pred) == pred[0]:
            if labels[index] == 0:
                matrix[0][0] += 1
            if labels[index] == 1:
                matrix[0][1] += 1
        elif np.amax(pred) == pred[1]:
            if labels[index] == 0:
                matrix[1][0] += 1
            if labels[index] == 1:
                matrix[1][1] += 1
    return matrix


def matrix_sum(A, B): 
    return [[A[0][0]+B[0][0], A[0][1]+B[0][1]],
            [A[1][0]+B[1][0], A[1][1]+B[1][1]]]


def get_accu(matrix):
    return float(matrix[0][0] + matrix[1][1])/ float(sum(matrix[0]) + sum(matrix[1]))


def get_MCC(matrix):
    TP, TN, FP, FN = float(matrix[0][0]), float(matrix[1][1]), float(matrix[0][1]), float(matrix[1][0])
    upper = TP * TN - FP * FN
    lower = (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)
    return upper / (lower**0.5 + 0.000000001)


def get_AD_risk(raw):
    x1, x2 = raw[0, :, :, :], raw[1, :, :, :]
    risk = np.exp(x2) / (np.exp(x1) + np.exp(x2))
    return risk


def read_json(config_file):
    with open(config_file) as config_buffer:
        config = json.loads(config_buffer.read())
    return config


def write_raw_score(f, preds, labels):
    preds = preds.data.cpu().numpy()
    labels = labels.data.cpu().numpy()
    for index, pred in enumerate(preds):
        label = str(labels[index])
        pred = "__".join(map(str, list(pred)))
        f.write(pred + '__' + label + '\n')


def write_raw_score_sk(f, preds, labels):
    for index, pred in enumerate(preds):
        label = str(labels[index])
        pred = "__".join(map(str, list(pred)))
        f.write(pred + '__' + label + '\n')


def read_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)
    filenames = [a[0] for a in your_list[1:]]
    labels = [0 if a[1]=='NL' else 1 for a in your_list[1:]]
    return filenames, labels


def read_csv_complete(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)
    filenames, labels, demors = [], [], []
    for line in your_list:
        try:
            demor = list(map(float, line[2:5]))
            gender = [0, 1] if demor[1] == 1 else [1, 0]
            demor = [(demor[0]-70.0)/10.0] + gender + [(demor[2]-27)/2]
            # demor = [demor[0]] + gender + demor[2:]
        except:
            continue
        filenames.append(line[0])
        label = 0 if line[1]=='NL' else 1
        labels.append(label)
        demors.append(demor)
    return filenames, labels, demors


def read_csv_complete_apoe(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)
    filenames, labels, demors = [], [], []
    for line in your_list:
        try:
            demor = list(map(float, line[2:6]))
            gender = [0, 1] if demor[1] == 1 else [1, 0]
            demor = [(demor[0] - 70.0) / 10.0] + gender + [(demor[2] - 27) / 2] + [demor[3]]
        except:
            continue
        filenames.append(line[0])
        label = 0 if line[1] == 'NL' else 1
        labels.append(label)
        demors.append(demor)
    return filenames, labels, demors


def data_split(repe_time):
    with open('./lookupcsv/ADNI.csv', 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)
    labels, train_valid, test = your_list[0:1], your_list[1:338], your_list[338:]
    for i in range(repe_time):
        random.shuffle(train_valid)
        folder = 'lookupcsv/exp{}/'.format(i)
        if not os.path.exists(folder):
            os.mkdir(folder) 
        with open(folder + 'train.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerows(labels + train_valid[:250])
        with open(folder + 'valid.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerows(labels + train_valid[250:])
        with open(folder + 'test.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerows(labels + test)


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed


def DPM_statistics(DPMs, Labels):
    shape = DPMs[0].shape[1:]
    voxel_number = shape[0] * shape[1] * shape[2]
    TP, FP, TN, FN = np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape)
    for label, DPM in zip(Labels, DPMs):
        risk_map = get_AD_risk(DPM)
        if label == 0:
            TN += (risk_map < 0.5).astype(np.int)
            FP += (risk_map >= 0.5).astype(np.int)
        elif label == 1:
            TP += (risk_map >= 0.5).astype(np.int)
            FN += (risk_map < 0.5).astype(np.int)
    tn = float("{0:.2f}".format(np.sum(TN) / voxel_number))
    fn = float("{0:.2f}".format(np.sum(FN) / voxel_number))
    tp = float("{0:.2f}".format(np.sum(TP) / voxel_number))
    fp = float("{0:.2f}".format(np.sum(FP) / voxel_number))
    matrix = [[tn, fn], [fp, tp]]
    count = len(Labels)
    TP, TN, FP, FN = TP.astype(np.float)/count, TN.astype(np.float)/count, FP.astype(np.float)/count, FN.astype(np.float)/count
    ACCU = TP + TN
    F1 = 2*TP/(2*TP+FP+FN)
    MCC = (TP*TN-FP*FN)/(np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))+0.00000001*np.ones(shape))
    return matrix, ACCU, F1, MCC


