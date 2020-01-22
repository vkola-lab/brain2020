import numpy as np
import json
import sys
import os
from utils import write_raw_score_sk
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from dataloader import MLP_Data
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier

def read_json(config_file):
    with open(config_file) as config_buffer:
        config = json.loads(config_buffer.read())
    return config

def gen_array(dataset, model):
    X, Y = [], []
    for i in range(len(dataset)):
        x, y, z = dataset[i]
        if model == 'A':
            x = x
        elif model == 'B':
            x = z
        elif model == 'C':
            x = np.concatenate((x, z))
        X.append(x)
        Y.append(y)
    return X, Y


def norm(List_X):
    norm_list = []
    for X in List_X:
        X = np.array(X)
        X = (X - X.mean()) / X.std()
        X = X.tolist()
        norm_list.append(X)
    return norm_list


def write_score(exp_idx, model, clf, X_test, y_test, X_NACC, y_NACC, X_AIBL, y_AIBL, X_FHS, y_FHS):
    for model in ['A', 'B', 'C']:
        if not os.path.exists('./checkpoint_dir/mlp_{}_exp{}/'.format(model, exp_idx)):
            os.mkdir('./checkpoint_dir/mlp_{}_exp{}/'.format(model, exp_idx))
    f = open('./checkpoint_dir/mlp_{}_exp{}/raw_score_test.txt'.format(model, exp_idx), 'w')
    write_raw_score_sk(f, clf.predict_proba(X_test), y_test)
    f.close()
    f = open('./checkpoint_dir/mlp_{}_exp{}/raw_score_NACC.txt'.format(model, exp_idx), 'w')
    write_raw_score_sk(f, clf.predict_proba(X_NACC), y_NACC)
    f.close()
    f = open('./checkpoint_dir/mlp_{}_exp{}/raw_score_AIBL.txt'.format(model, exp_idx), 'w')
    write_raw_score_sk(f, clf.predict_proba(X_AIBL), y_AIBL)
    f.close()
    f = open('./checkpoint_dir/mlp_{}_exp{}/raw_score_FHS.txt'.format(model, exp_idx), 'w')
    write_raw_score_sk(f, clf.predict_proba(X_FHS), y_FHS)
    f.close()

def mlp_A_train(exp_idx, repe_time, roi_threshold, alpha, batch_size, init_lr, max_iter, hidden_size, accu):
    X_train, y_train = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'train', roi_threshold, seed=1000), 'A')
    X_valid, y_valid = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'valid', roi_threshold, seed=1000), 'A')
    X_test, y_test = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'test', roi_threshold, seed=1000), 'A')
    X_NACC, y_NACC = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'NACC', roi_threshold, seed=1000), 'A')
    X_AIBL, y_AIBL = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'AIBL', roi_threshold, seed=1000), 'A')
    X_FHS, y_FHS = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'FHS', roi_threshold, seed=1000), 'A')

    #X_train, X_valid, X_test, X_NACC, X_AIBL, X_FHS = norm([X_train, X_valid, X_test, X_NACC, X_AIBL, X_FHS])

    for idx in range(repe_time):
        clf = MLPClassifier(hidden_layer_sizes=(hidden_size), alpha=alpha, batch_size=batch_size, learning_rate_init=init_lr, max_iter=max_iter, tol=0)
        clf.fit(X_train + X_valid, y_train + y_valid)
        write_score(exp_idx, 'A', clf, X_test, y_test, X_NACC, y_NACC, X_AIBL, y_AIBL, X_FHS, y_FHS)
        accu['A']['test'].append(accuracy_score(y_test, clf.predict(X_test)))
        accu['A']['NACC'].append(accuracy_score(y_NACC, clf.predict(X_NACC)))
        accu['A']['AIBL'].append(accuracy_score(y_AIBL, clf.predict(X_AIBL)))
        accu['A']['FHS'].append(accuracy_score(y_FHS, clf.predict(X_FHS)))

def mlp_B_train(exp_idx, repe_time, roi_threshold, accu):
    X_train, y_train = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'train', roi_threshold, seed=1000), 'B')
    X_valid, y_valid = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'valid', roi_threshold, seed=1000), 'B')
    X_test, y_test = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'test', roi_threshold, seed=1000), 'B')
    X_NACC, y_NACC = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'NACC', roi_threshold, seed=1000), 'B')
    X_AIBL, y_AIBL = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'AIBL', roi_threshold, seed=1000), 'B')
    X_FHS, y_FHS = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'FHS', roi_threshold, seed=1000), 'B')

    for idx in range(repe_time):
        clf = MLPClassifier(hidden_layer_sizes=(100), max_iter=1000)
        clf.fit(X_train + X_valid, y_train + y_valid)
        write_score(exp_idx, 'B', clf, X_test, y_test, X_NACC, y_NACC, X_AIBL, y_AIBL, X_FHS, y_FHS)
        accu['B']['test'].append(accuracy_score(y_test, clf.predict(X_test)))
        accu['B']['NACC'].append(accuracy_score(y_NACC, clf.predict(X_NACC)))
        accu['B']['AIBL'].append(accuracy_score(y_AIBL, clf.predict(X_AIBL)))
        accu['B']['FHS'].append(accuracy_score(y_FHS, clf.predict(X_FHS)))

def mlp_C_train(exp_idx, repe_time, roi_threshold, accu):
    X_train, y_train = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'train', roi_threshold, seed=1000), 'C')
    X_valid, y_valid = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'valid', roi_threshold, seed=1000), 'C')
    X_test, y_test = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'test', roi_threshold, seed=1000), 'C')
    X_NACC, y_NACC = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'NACC', roi_threshold, seed=1000), 'C')
    X_AIBL, y_AIBL = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'AIBL', roi_threshold, seed=1000), 'C')
    X_FHS, y_FHS = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'FHS', roi_threshold, seed=1000), 'C')

    for idx in range(repe_time):
        clf = MLPClassifier(hidden_layer_sizes=(100), max_iter=1000)
        clf.fit(X_train + X_valid, y_train + y_valid)
        write_score(exp_idx, 'C', clf, X_test, y_test, X_NACC, y_NACC, X_AIBL, y_AIBL, X_FHS, y_FHS)
        accu['C']['test'].append(accuracy_score(y_test, clf.predict(X_test)))
        accu['C']['NACC'].append(accuracy_score(y_NACC, clf.predict(X_NACC)))
        accu['C']['AIBL'].append(accuracy_score(y_AIBL, clf.predict(X_AIBL)))
        accu['C']['FHS'].append(accuracy_score(y_FHS, clf.predict(X_FHS)))

def tune_A(roi=0.6, alpha=0.0, batch_size='auto', lr=0.01, max_iter=1500, hidden_size=5000):
    accu = {'A': {'test': [], 'NACC': [], 'AIBL': [], 'FHS': []}}
    for exp_idx in range(5):
        mlp_A_train(exp_idx, 3, roi, alpha, batch_size, lr, max_iter, hidden_size, accu)
    print('roi {} alpha {} batch_size {} lr {}, max_iter {} hidden {}'.format(roi, alpha, batch_size, lr, max_iter, hidden_size))
    print('ADNI test accuracy ',
          'A {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['A']['test'])), float(np.std(accu['A']['test']))))
    print('NACC test accuracy ',
          'A {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['A']['NACC'])), float(np.std(accu['A']['NACC']))))
    print('AIBL test accuracy ',
          'A {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['A']['AIBL'])), float(np.std(accu['A']['AIBL']))))
    print('FHS test accuracy  ',
          'A {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['A']['FHS'])), float(np.std(accu['A']['FHS']))))


def mlp():
    accu = {'A':{'test':[], 'NACC':[], 'AIBL':[], 'FHS':[]}, \
            'B':{'test':[], 'NACC':[], 'AIBL':[], 'FHS':[]}, \
            'C':{'test':[], 'NACC':[], 'AIBL':[], 'FHS':[]}}

    for exp_idx in range(5):
        mlp_A_train(exp_idx, 1, 0.5, accu)
        mlp_B_train(exp_idx, 1, 0.5, accu)
        mlp_C_train(exp_idx, 1, 0.5, accu)

    print('ADNI test accuracy ',
          'A {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['A']['test'])), float(np.std(accu['A']['test']))), \
          'B {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['B']['test'])), float(np.std(accu['B']['test']))), \
          'C {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['C']['test'])), float(np.std(accu['C']['test']))))
    print('NACC test accuracy ',
          'A {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['A']['NACC'])), float(np.std(accu['A']['NACC']))), \
          'B {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['B']['NACC'])), float(np.std(accu['B']['NACC']))), \
          'C {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['C']['NACC'])), float(np.std(accu['C']['NACC']))))
    print('AIBL test accuracy ',
          'A {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['A']['AIBL'])), float(np.std(accu['A']['AIBL']))), \
          'B {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['B']['AIBL'])), float(np.std(accu['B']['AIBL']))), \
          'C {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['C']['AIBL'])), float(np.std(accu['C']['AIBL']))))
    print('FHS test accuracy  ',
          'A {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['A']['FHS'])), float(np.std(accu['A']['FHS']))), \
          'B {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['B']['FHS'])), float(np.std(accu['B']['FHS']))), \
          'C {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['C']['FHS'])), float(np.std(accu['C']['FHS']))))


def mlp_A_thres(exp_idx, roi_threshold):
    X_train, y_train = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'train', roi_threshold, seed=1000), 'A')
    X_valid, y_valid = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'valid', roi_threshold, seed=1000), 'A')
    clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
    clf.fit(X_train, y_train)
    return accuracy_score(y_valid, clf.predict(X_valid))


def find_MCC_threshold():
    for roi_val in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        accu = []
        for exp_idx in range(5):
            accu.append(mlp_A_thres(exp_idx, roi_val))
        print('Roi threshold {0:.1f} {1:.4f}+/-{2:.4f}'.format(roi_val, float(np.mean(accu)), float(np.std(accu))))


def hypertune(config, model):
    accu = []
    for exp_idx in range(5):
        for _ in range(2):
            setting = read_json(config)
            X_train, y_train = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'train', setting["roi_threshold"], seed=1000), model)
            X_valid, y_valid = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'valid', setting["roi_threshold"], seed=1000), model)
            clf = MLPClassifier(max_iter            = setting['train_epochs'],
                                hidden_layer_sizes  = (setting['fil_num1'], setting['fil_num2']),
                                learning_rate_init  = setting['learning_rate'],
                                alpha               = setting['alpha'],
                                batch_size          = setting['batch_size'])
            clf.fit(X_train, y_train)
            accu.append(accuracy_score(y_valid, clf.predict(X_valid)))
    print('$' + str(1 - float(np.mean(accu))) + '$$')


if __name__ == "__main__":
    # find_MCC_threshold()
    # mlp()
    # filename = sys.argv[1]
    # hypertune(filename, 'A')


    # for roi in [-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    # for size in [4000, 5000, 6000, 7000]:
    # for init_lr in [1, 0.1, 0.001, 0.0001, 0.00001]:

    # for exp_idx in range(5):
    #     accu = {'A': {'test': [], 'NACC': [], 'AIBL': [], 'FHS': []}}
    #     mlp_A_train(exp_idx, 1, 0.6, 0.0, 'auto', 0.01, 1500, 100, accu)
    #     print(accu)



    # for batch_size in [4, 8, 16, 32, 64, 128, 'auto']:
    # for lr in [0.1, 0.01, 0.001, 0.0001]:
    # for roi in [0.6, 0.5, 0.4, 0.3]:
    tune_A(roi=0.6, alpha=0.0, batch_size='auto', lr=0.01, max_iter=1500, hidden_size=100)


"""
Roi threshold 0.0 0.7724+/-0.0746
Roi threshold 0.1 0.8023+/-0.0592
Roi threshold 0.2 0.7793+/-0.0312
Roi threshold 0.3 0.7701+/-0.0638
Roi threshold 0.4 0.7954+/-0.0676
Roi threshold 0.5 0.7954+/-0.0656
Roi threshold 0.6 0.8230+/-0.0580
"""
