import numpy as np
import json
import sys
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

def mlp_train_test(config_A, config_B, config_C):
    accu = {'A':{}, 'B':{}, 'C':{}}
    setting = {}

    setting['A'] = read_json(config_A)
    setting['B'] = read_json(config_B)
    setting['C'] = read_json(config_C)

    #################################################################
    #################################################################
    model = 'A'

    X_train, y_train = gen_array(
        MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'train', setting[model]['roi_threshold'], seed=1000), model)
    X_valid, y_valid = gen_array(
        MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'valid', setting[model]['roi_threshold'], seed=1000), model)
    X_test, y_test = gen_array(
        MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'test', setting[model]['roi_threshold'], seed=1000), model)
    X_NACC, y_NACC = gen_array(
        MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'NACC', setting[model]['roi_threshold'], seed=1000), model)
    X_AIBL, y_AIBL = gen_array(
        MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'AIBL', setting[model]['roi_threshold'], seed=1000), model)
    X_FHS, y_FHS = gen_array(
        MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'FHS', setting[model]['roi_threshold'], seed=1000), model)

    clf = MLPClassifier(max_iter            = setting[model]['train_epochs'],
                        hidden_layer_sizes  = (setting[model]['fil_num1'], setting[model]['fil_num2']),
                        learning_rate_init  = setting[model]['learning_rate'],
                        alpha               = setting[model]['alpha'],
                        batch_size          = setting[model]['batch_size'])
    clf = MLPClassifier(hidden_layer_sizes = (20,),
                        max_iter = 2000)

    clf.fit(X_train+X_valid, y_train+y_valid)

    y_test_pred = clf.predict(X_test)
    y_NACC_pred = clf.predict(X_NACC)
    y_AIBL_pred = clf.predict(X_AIBL)
    y_FHS_pred = clf.predict(X_FHS)

    accu[model]['test'] = accuracy_score(y_test, y_test_pred)
    accu[model]['NACC'] = accuracy_score(y_NACC, y_NACC_pred)
    accu[model]['AIBL'] = accuracy_score(y_AIBL, y_AIBL_pred)
    accu[model]['FHS']  = accuracy_score(y_FHS,  y_FHS_pred)

    #################################################################
    #################################################################
    model = 'B'
    X_train, y_train = gen_array(
        MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'train', setting[model]['roi_threshold'], seed=1000),
        model)
    X_valid, y_valid = gen_array(
        MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'valid', setting[model]['roi_threshold'], seed=1000),
        model)
    X_test, y_test = gen_array(
        MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'test', setting[model]['roi_threshold'], seed=1000),
        model)
    X_NACC, y_NACC = gen_array(
        MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'NACC', setting[model]['roi_threshold'], seed=1000),
        model)
    X_AIBL, y_AIBL = gen_array(
        MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'AIBL', setting[model]['roi_threshold'], seed=1000),
        model)
    X_FHS, y_FHS = gen_array(
        MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'FHS', setting[model]['roi_threshold'], seed=1000),
        model)

    clf = MLPClassifier(max_iter=setting[model]['train_epochs'],
                        hidden_layer_sizes=setting[model]['fil_num'],
                        learning_rate_init=setting[model]['learning_rate'],
                        alpha=setting[model]['alpha'],
                        batch_size=setting[model]['batch_size'])
    clf = MLPClassifier(hidden_layer_sizes = (200,),
                        max_iter=2000)

    clf.fit(X_train + X_valid, y_train + y_valid)

    y_test_pred = clf.predict(X_test)
    y_NACC_pred = clf.predict(X_NACC)
    y_AIBL_pred = clf.predict(X_AIBL)
    y_FHS_pred = clf.predict(X_FHS)

    accu[model]['test'] = accuracy_score(y_test, y_test_pred)
    accu[model]['NACC'] = accuracy_score(y_NACC, y_NACC_pred)
    accu[model]['AIBL'] = accuracy_score(y_AIBL, y_AIBL_pred)
    accu[model]['FHS'] = accuracy_score(y_FHS, y_FHS_pred)

    #################################################################
    #################################################################
    model = 'C'
    X_train, y_train = gen_array(
        MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'train', setting[model]['roi_threshold'], seed=1000),
        model)
    X_valid, y_valid = gen_array(
        MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'valid', setting[model]['roi_threshold'], seed=1000),
        model)
    X_test, y_test = gen_array(
        MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'test', setting[model]['roi_threshold'], seed=1000),
        model)
    X_NACC, y_NACC = gen_array(
        MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'NACC', setting[model]['roi_threshold'], seed=1000),
        model)
    X_AIBL, y_AIBL = gen_array(
        MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'AIBL', setting[model]['roi_threshold'], seed=1000),
        model)
    X_FHS, y_FHS = gen_array(
        MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'FHS', setting[model]['roi_threshold'], seed=1000),
        model)

    clf = MLPClassifier(max_iter=setting[model]['train_epochs'],
                        hidden_layer_sizes=setting[model]['fil_num'],
                        learning_rate_init=setting[model]['learning_rate'],
                        alpha=setting[model]['alpha'],
                        batch_size=setting[model]['batch_size'])
    clf = MLPClassifier(hidden_layer_sizes = (200,),
                        max_iter=2000)

    clf.fit(X_train + X_valid, y_train + y_valid)

    y_test_pred = clf.predict(X_test)
    y_NACC_pred = clf.predict(X_NACC)
    y_AIBL_pred = clf.predict(X_AIBL)
    y_FHS_pred = clf.predict(X_FHS)

    accu[model]['test'] = accuracy_score(y_test, y_test_pred)
    accu[model]['NACC'] = accuracy_score(y_NACC, y_NACC_pred)
    accu[model]['AIBL'] = accuracy_score(y_AIBL, y_AIBL_pred)
    accu[model]['FHS'] = accuracy_score(y_FHS, y_FHS_pred)

    print('ADNI test accuracy ', 'A %.4f '%accu['A']['test'],  'B %.4f '%accu['B']['test'], 'C %.4f '%accu['C']['test'])
    print('NACC test accuracy ', 'A %.4f '%accu['A']['NACC'],  'B %.4f '%accu['B']['NACC'], 'C %.4f '%accu['C']['NACC'])
    print('AIBL test accuracy ', 'A %.4f '%accu['A']['AIBL'],  'B %.4f '%accu['B']['AIBL'], 'C %.4f '%accu['C']['AIBL'])
    print(' FHS test accuracy ', 'A %.4f '%accu['A']['FHS'],   'B %.4f '%accu['B']['FHS'],  'C %.4f '%accu['C']['FHS'])

def simple_mlp_train_test():
    accu = {'A':{}, 'B':{}, 'C':{}}
    roi_threshold = 0.5

    for model in ['A', 'B', 'C']:

        X_train, y_train = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'train', roi_threshold, seed=1000), model)
        X_valid, y_valid = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'valid', roi_threshold, seed=1000), model)
        X_test, y_test = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'test', roi_threshold, seed=1000), model)
        X_NACC, y_NACC = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'NACC', roi_threshold, seed=1000), model)
        X_AIBL, y_AIBL = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'AIBL', roi_threshold, seed=1000), model)
        X_FHS, y_FHS = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'FHS', roi_threshold, seed=1000), model)

        clf = MLPClassifier(hidden_layer_sizes = (100),
                            max_iter = 1000)

        clf.fit(X_train+X_valid, y_train+y_valid)

        y_test_pred = clf.predict(X_test)
        y_NACC_pred = clf.predict(X_NACC)
        y_AIBL_pred = clf.predict(X_AIBL)
        y_FHS_pred = clf.predict(X_FHS)

        accu[model]['test'] = accuracy_score(y_test, y_test_pred)
        accu[model]['NACC'] = accuracy_score(y_NACC, y_NACC_pred)
        accu[model]['AIBL'] = accuracy_score(y_AIBL, y_AIBL_pred)
        accu[model]['FHS']  = accuracy_score(y_FHS,  y_FHS_pred)

    print('ADNI test accuracy ', 'A %.4f '%accu['A']['test'],  'B %.4f '%accu['B']['test'], 'C %.4f '%accu['C']['test'])
    print('NACC test accuracy ', 'A %.4f '%accu['A']['NACC'],  'B %.4f '%accu['B']['NACC'], 'C %.4f '%accu['C']['NACC'])
    print('AIBL test accuracy ', 'A %.4f '%accu['A']['AIBL'],  'B %.4f '%accu['B']['AIBL'], 'C %.4f '%accu['C']['AIBL'])
    print(' FHS test accuracy ', 'A %.4f '%accu['A']['FHS'],   'B %.4f '%accu['B']['FHS'],  'C %.4f '%accu['C']['FHS'])


def write_score(exp_idx, model, clf, X_test, y_test, X_NACC, y_NACC, X_AIBL, y_AIBL, X_FHS, y_FHS):
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

def mlp_A_train(exp_idx, repe_time, roi_threshold, accu):
    X_train, y_train = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'train', roi_threshold, seed=1000), 'A')
    X_valid, y_valid = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'valid', roi_threshold, seed=1000), 'A')
    X_test, y_test = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'test', roi_threshold, seed=1000), 'A')
    X_NACC, y_NACC = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'NACC', roi_threshold, seed=1000), 'A')
    X_AIBL, y_AIBL = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'AIBL', roi_threshold, seed=1000), 'A')
    X_FHS, y_FHS = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'FHS', roi_threshold, seed=1000), 'A')

    for idx in range(repe_time):
        clf = MLPClassifier(hidden_layer_sizes=(200, 100),
                            max_iter=440,
                            alpha=0.379,
                            batch_size=128,
                            learning_rate_init=0.000629
                            )
        clf = MLPClassifier(max_iter=1000)
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

def mlp():
    accu = {'A':{'test':[], 'NACC':[], 'AIBL':[], 'FHS':[]}, \
            'B':{'test':[], 'NACC':[], 'AIBL':[], 'FHS':[]}, \
            'C':{'test':[], 'NACC':[], 'AIBL':[], 'FHS':[]}}

    for exp_idx in range(3):
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

def hypertune(exp_idx, model, config):
    setting = read_json(config)
    X_train, y_train = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'train', setting["roi_threshold"], seed=1000), model)
    X_valid, y_valid = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'valid', setting["roi_threshold"], seed=1000), model)
    clf = MLPClassifier(max_iter            = setting['train_epochs'],
                        hidden_layer_sizes  = (setting['fil_num1'], setting['fil_num2']),
                        learning_rate_init  = setting['learning_rate'],
                        alpha               = setting['alpha'],
                        batch_size          = setting['batch_size'])
    clf.fit(X_train, y_train)
    y_valid_pred = clf.predict(X_valid)
    print('$' + str(1 - accuracy_score(y_valid, y_valid_pred)) + '$$')

if __name__ == "__main__":
    # filename = sys.argv[1]
    # hypertune(1, 'A', filename)
    # mlp_train_test('mlp_config_A.json', 'mlp_config_B.json', 'mlp_config_C.json')
    # simple_mlp_train_test()
    mlp()