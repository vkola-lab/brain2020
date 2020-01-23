from utils_stat import get_roc_info, get_pr_info, calc_neurologist_statistics, read_raw_score
import matplotlib.pyplot as plt
from utils_plot import plot_curve, plot_legend, plot_neorologist
from time import time
import numpy as np

def confusion_matrix(labels, scores):
    matrix = [[0, 0], [0, 0]]
    for label, pred in zip(labels, scores):
        if pred < 0.5:
            if label == 0:
                matrix[0][0] += 1
            if label == 1:
                matrix[0][1] += 1
        else:
            if label == 0:
                matrix[1][0] += 1
            if label == 1:
                matrix[1][1] += 1
    return matrix


def get_metrics(matrix):
    TP, FP, TN, FN = matrix[1][1], matrix[1][0], matrix[0][0], matrix[0][1]
    TP, FP, TN, FN = float(TP), float(FP), float(TN), float(FN)
    ACCU = (TP + TN) / (TP + TN + FP + FN)
    Sens = TP / (TP + FN + 0.0000001)
    Spec = TN / (TN + FP + 0.0000001)
    F1 = 2*TP/(2*TP+FP+FN)
    MCC = (TP*TN-FP*FN)/((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)+0.00000001)**0.5
    return ACCU, Sens, Spec, F1, MCC


def stat_metric(matrices):
    Accu, Sens, Spec, F1, MCC = [], [], [], [], []
    for matrix in matrices:
        accu, sens, spec, f1, mcc = get_metrics(matrix)
        Accu.append(accu)
        Sens.append(sens)
        Spec.append(spec)
        F1.append(f1)
        MCC.append(mcc)
    print('Accu {0:.4f}+/-{1:.4f}'.format(float(np.mean(Accu)), float(np.std(Accu))))
    print('Sens {0:.4f}+/-{1:.4f}'.format(float(np.mean(Sens)), float(np.std(Sens))))
    print('Spec {0:.4f}+/-{1:.4f}'.format(float(np.mean(Spec)), float(np.std(Spec))))
    print('F1   {0:.4f}+/-{1:.4f}'.format(float(np.mean(F1)),   float(np.std(F1))))
    print('MCC  {0:.4f}+/-{1:.4f}'.format(float(np.mean(MCC)), float(np.std(MCC))))

if __name__ == "__main__":
    Matrix = []
    for i in range(10):
        labels, scores = read_raw_score('../checkpoint_dir/Vol_RF/raw_score_{}.txt'.format(i))
        Matrix.append(confusion_matrix(labels, scores))
    stat_metric(Matrix)



