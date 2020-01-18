import numpy as np
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

exp_idx = 1
roi_threshold = 0.5

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


def mlp_train_test():
    accu = {'A':{}, 'B':{}, 'C':{}}

    for model in ['A', 'B', 'C']:
        X_train, y_train = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'train', roi_threshold, seed=1000), model)
        X_valid, y_valid = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'valid', roi_threshold, seed=1000), model)
        X_test, y_test = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'test', roi_threshold, seed=1000), model)
        X_NACC, y_NACC = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'NACC', roi_threshold, seed=1000), model)
        X_AIBL, y_AIBL = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'AIBL', roi_threshold, seed=1000), model)
        X_FHS, y_FHS = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'FHS', roi_threshold, seed=1000), model)

        clf = MLPClassifier(max_iter=1000)

        clf.fit(X_train+X_valid, y_train+y_valid)

        y_test_pred = clf.predict(X_test)
        y_NACC_pred = clf.predict(X_NACC)
        y_AIBL_pred = clf.predict(X_AIBL)
        y_FHS_pred = clf.predict(X_FHS)

        test_matrix = confusion_matrix(y_test, y_test_pred)
        NACC_matrix = confusion_matrix(y_NACC, y_NACC_pred)
        AIBL_matrix = confusion_matrix(y_AIBL, y_AIBL_pred)
        FHS_matrix = confusion_matrix(y_FHS, y_FHS_pred)

        accu[model]['test'] = accuracy_score(y_test, y_test_pred)
        accu[model]['NACC'] = accuracy_score(y_NACC, y_NACC_pred)
        accu[model]['AIBL'] = accuracy_score(y_AIBL, y_AIBL_pred)
        accu[model]['FHS']  = accuracy_score(y_FHS,  y_FHS_pred)

    print('ADNI test accuracy ', 'A %.4f '%accu['A']['test'],  'B %.4f '%accu['B']['test'], 'C %.4f '%accu['C']['test'])
    print('NACC test accuracy ', 'A %.4f '%accu['A']['NACC'],  'B %.4f '%accu['B']['NACC'], 'C %.4f '%accu['C']['NACC'])
    print('AIBL test accuracy ', 'A %.4f '%accu['A']['AIBL'],  'B %.4f '%accu['B']['AIBL'], 'C %.4f '%accu['C']['AIBL'])
    print(' FHS test accuracy ', 'A %.4f '%accu['A']['FHS'],   'B %.4f '%accu['B']['FHS'],  'C %.4f '%accu['C']['FHS'])


def hypertune(exp_idx, model, config):
    X_train, y_train = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'train', roi_threshold, seed=1000), model)
    X_valid, y_valid = gen_array(MLP_Data('./DPMs/fcn_exp{}/'.format(exp_idx), exp_idx, 'valid', roi_threshold, seed=1000), model)
    clf = MLPClassifier(**config)
    clf.fit(X_train, y_train)
    y_valid_pred = clf.predict(X_valid)
    return accuracy_score(y_valid, y_valid_pred)

if __name__ == "__main__":

