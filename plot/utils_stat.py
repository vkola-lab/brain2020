# -*- coding: utf-8 -*-
"""
Created on Wed May 29 23:06:43 2019
@author: Iluva
"""
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from scipy import interp
from collections import defaultdict
import numpy as np
import math
import csv

def softmax(a, b):
    Max = max([a, b])
    a, b = a-Max, b-Max
    return math.exp(b) / (math.exp(a) + math.exp(b))

def read_raw_score(txt_file):
    labels, scores = [], []
    with open(txt_file, 'r') as f:
        for line in f:
            nl, ad, label = map(float, line.strip('\n').split('__'))
            scores.append(softmax(nl, ad))
            labels.append(int(label))
    return np.array(labels), np.array(scores)

def pr_interp(rc_, rc, pr):
    pr_ = np.zeros_like(rc_)
    locs = np.searchsorted(rc, rc_)
    for idx, loc in enumerate(locs):
        l = loc - 1
        r = loc
        r1 = rc[l] if l > -1 else 0
        r2 = rc[r] if r < len(rc) else 1
        p1 = pr[l] if l > -1 else 1
        p2 = pr[r] if r < len(rc) else 0

        t1 = (1 - p2) * r2 / p2 / (r2 - r1) if p2 * (r2 - r1) > 1e-16 else (1 - p2) * r2 / 1e-16
        t2 = (1 - p1) * r1 / p1 / (r2 - r1) if p1 * (r2 - r1) > 1e-16 else (1 - p1) * r1 / 1e-16
        t3 = (1 - p1) * r1 / p1 if p1 > 1e-16 else (1 - p1) * r1 / 1e-16

        a = 1 + t1 - t2
        b = t3 - t1 * r1 + t2 * r1
        pr_[idx] = rc_[idx] / (a * rc_[idx] + b)
    return pr_


def calc_performance_statistics(all_scores, y):
    statistics = {}
    y_pred_all = np.argmax(all_scores, axis=2)

    statistics = defaultdict(list)
    for y_pred in y_pred_all:
        TN, FP, FN, TP = confusion_matrix(y, y_pred).ravel()
        N = TN + TP + FN + FP
        S = (TP + FN) / N
        P = (TP + FP) / N
        acc = (TN + TP) / N
        sen = TP / (TP + FN)
        spc = TN / (TN + FP)
        prc = TP / (TP + FP)
        f1s = 2 * (prc * sen) / (prc + sen)
        mcc = (TP / N - S * P) / np.sqrt(P * S * (1 - S) * (1 - P))

        statistics['confusion_matrix'].append(confusion_matrix(y, y_pred))
        statistics['accuracy'].append(acc)
        statistics['sensitivity'].append(sen)
        statistics['specificity'].append(spc)
        statistics['precision'].append(prc)
        statistics['f1_score'].append(f1s)
        statistics['MCC'].append(mcc)

    return statistics


def get_roc_info(y, y_score_list):
    fpr_pt = np.linspace(0, 1, 1001)
    tprs, aucs = [], []
    for y_score in y_score_list:
        fpr, tpr, _ = roc_curve(y_true=y, y_score=y_score, drop_intermediate=True)
        tprs.append(interp(fpr_pt, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(auc(fpr, tpr))
    tprs_mean = np.mean(tprs, axis=0)
    tprs_std = np.std(tprs, axis=0)
    tprs_upper = np.minimum(tprs_mean + tprs_std, 1)
    tprs_lower = np.maximum(tprs_mean - tprs_std, 0)
    auc_mean = auc(fpr_pt, tprs_mean)
    auc_std = np.std(aucs)
    auc_std = 1 - auc_mean if auc_mean + auc_std > 1 else auc_std

    rslt = {'xs': fpr_pt,
            'ys_mean': tprs_mean,
            'ys_upper': tprs_upper,
            'ys_lower': tprs_lower,
            'auc_mean': auc_mean,
            'auc_std': auc_std}

    return rslt


def get_pr_info(y, y_score_list):
    rc_pt = np.linspace(0, 1, 1001)
    rc_pt[0] = 1e-16
    prs = []
    aps = []
    for y_score in y_score_list:
        pr, rc, _ = precision_recall_curve(y_true=y, probas_pred=y_score)
        aps.append(average_precision_score(y_true=y, y_score=y_score))
        pr, rc = pr[::-1], rc[::-1]
        prs.append(pr_interp(rc_pt, rc, pr))

    prs_mean = np.mean(prs, axis=0)
    prs_std = np.std(prs, axis=0)
    prs_upper = np.minimum(prs_mean + prs_std, 1)
    prs_lower = np.maximum(prs_mean - prs_std, 0)
    aps_mean = np.mean(aps)
    aps_std = np.std(aps)
    aps_std = 1 - aps_mean if aps_mean + aps_std > 1 else aps_std

    rslt = {'xs': rc_pt,
            'ys_mean': prs_mean,
            'ys_upper': prs_upper,
            'ys_lower': prs_lower,
            'auc_mean': aps_mean,
            'auc_std': aps_std}

    return rslt


def calc_neurologist_statistics(y, y_pred_list):
    rslt, sens, spcs, prcs = {}, [], [], []
    for i, y_pred in enumerate(y_pred_list):
        TN, FP, FN, TP = confusion_matrix(y, y_pred).ravel()
        sens.append(TP / (TP + FN))
        spcs.append(TN / (TN + FP))
        prcs.append(TP / (TP + FP))
        rslt['neorologist_{}'.format(i)] = {'sensitivity': sens[-1], 'specificity': spcs[-1], 'precision': prcs[-1]}
    rslt['mean'] = {'sensitivity': np.mean(sens), 'specificity': np.mean(spcs), 'precision': np.mean(prcs)}
    rslt['std'] = {'sensitivity': np.std(sens), 'specificity': np.std(spcs), 'precision': np.std(prcs)}
    return rslt


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