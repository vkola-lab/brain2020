from utils_stat import get_roc_info, get_pr_info, load_neurologist_data, calc_neurologist_statistics, read_raw_score
import csv
import sys
sys.path.insert(1, '../')
#from utils import read_csv_complete
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

font = {'size':22}
matplotlib.rc('font', **font)

def read_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)
    filenames = [a[0] for a in your_list[1:]]
    ages = [a[2] for a in your_list[1:]]
    return ages



def get_type(preds, labels):
    pred_type = []
    for index, pred in enumerate(preds):
        if pred > 0.5:
            if labels[index] == 0:
                pred_type.append('FP')
            if labels[index] == 1:
                pred_type.append('TP')
        else:
            if labels[index] == 0:
                pred_type.append('TN')
            if labels[index] == 1:
                pred_type.append('FN')
    return pred_type

"""
ADNI_test MLPA
FHS, NACC, AIBL, MLPA
"""
csv_file = '../lookupcsv/'

fig, axs = plt.subplots(2, 2)

# collect essentials for plot
for i, ds in enumerate(['test', 'AIBL', 'FHS', 'NACC']):
    x, y = i>>1, i%2
    ax = axs[x, y]
    csv_file = '../lookupcsv/{}.csv'.format(ds)
    if ds == 'test':
        csv_file = '../lookupcsv/exp0/{}.csv'.format(ds)
    ages = read_csv(csv_file)
    type_list = []
    for exp_idx in range(5):
        for repe_idx in range(3):
            if ds == 'FHS':
                labels, scores = read_raw_score('../checkpoint_dir/mlp_{}_exp{}/raw_score_{}_Full_{}.txt'.format('A', exp_idx, ds, repe_idx))
            else:
                labels, scores = read_raw_score('../checkpoint_dir/mlp_{}_exp{}/raw_score_{}_{}.txt'.format('A', exp_idx, ds, repe_idx))
            type_list += get_type(scores, labels)


    ages = list(map(int, ages*15))
    TPs = [ages[i] for i in range(len(ages)) if type_list[i] == 'TP']
    TNs = [ages[i] for i in range(len(ages)) if type_list[i] == 'TN']
    FPs = [ages[i] for i in range(len(ages)) if type_list[i] == 'FP']
    FNs = [ages[i] for i in range(len(ages)) if type_list[i] == 'FN']
    data = [TPs, TNs, FPs, FNs]
    ax.set_title(ds)
    ax.boxplot(data)

    ax.set_xticks([1,2,3,4])
    ax.set_xticklabels(['TP', 'TN', 'FP', 'FN'])
    ax.set_yticks([50, 70, 90, 110])
    ax.set_ylabel('Age')


fig.set_size_inches(15, 8)
plt.savefig('./Figure S5', dpi=300)
