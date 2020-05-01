from model_wrapper import MLP_Wrapper_A, MLP_Wrapper_B, MLP_Wrapper_C, MLP_Wrapper_D, MLP_Wrapper_E, MLP_Wrapper_F
from utils import read_json
import numpy as np

def mlp_A_train(exp_idx, repe_time, accu, config):
    # mlp model build on features selected from disease probability maps (DPMs) generated from FCN
    mlp_setting = config
    for i in range(repe_time):
        mlp = MLP_Wrapper_A(imbalan_ratio=mlp_setting['imbalan_ratio'],
                            fil_num=mlp_setting['fil_num'],
                            drop_rate=mlp_setting['drop_rate'],
                            batch_size=mlp_setting['batch_size'],
                            balanced=mlp_setting['balanced'],
                            roi_threshold=mlp_setting['roi_threshold'],
                            roi_count=mlp_setting['roi_count'],
                            choice=mlp_setting['choice'],
                            exp_idx=exp_idx,
                            seed=seed+i,
                            model_name='mlp_A',
                            metric='accuracy')
        mlp.train(lr=mlp_setting['learning_rate'],
                  epochs=mlp_setting['train_epochs'])
        accu_test, accu_AIBL, accu_NACC, accu_FHS = mlp.test(i)[2:]
        accu['A']['test'].append(accu_test)
        accu['A']['NACC'].append(accu_NACC)
        accu['A']['AIBL'].append(accu_AIBL)
        accu['A']['FHS'].append(accu_FHS)


def mlp_B_train(exp_idx, repe_time, accu, config):
    # mlp build on non-imaging features, including age, gender, MMSE
    mlp_setting = config
    for i in range(repe_time):
        mlp = MLP_Wrapper_B(imbalan_ratio=mlp_setting['imbalan_ratio'],
                            fil_num=mlp_setting['fil_num'],
                            drop_rate=mlp_setting['drop_rate'],
                            batch_size=mlp_setting['batch_size'],
                            balanced=mlp_setting['balanced'],
                            roi_threshold=mlp_setting['roi_threshold'],
                            roi_count=mlp_setting['roi_count'],
                            choice=mlp_setting['choice'],
                            exp_idx=exp_idx,
                            seed=seed+i,
                            model_name='mlp_B_BN',
                            metric='accuracy')
        mlp.train(lr=mlp_setting['learning_rate'],
                  epochs=mlp_setting['train_epochs'])
        accu_test, accu_AIBL, accu_NACC, accu_FHS = mlp.test(i)[2:]
        accu['B']['test'].append(accu_test)
        accu['B']['NACC'].append(accu_NACC)
        accu['B']['AIBL'].append(accu_AIBL)
        accu['B']['FHS'].append(accu_FHS)


def mlp_C_train(exp_idx, repe_time, accu, config):
    # mlp build on combined features of mlp_A and mlp_B
    mlp_setting = config
    for i in range(repe_time):
        mlp = MLP_Wrapper_C(imbalan_ratio=mlp_setting['imbalan_ratio'],
                            fil_num=mlp_setting['fil_num'],
                            drop_rate=mlp_setting['drop_rate'],
                            batch_size=mlp_setting['batch_size'],
                            balanced=mlp_setting['balanced'],
                            roi_threshold=mlp_setting['roi_threshold'],
                            roi_count=mlp_setting['roi_count'],
                            choice=mlp_setting['choice'],
                            exp_idx=exp_idx,
                            seed=seed+i,
                            model_name='mlp_C_BN',
                            metric='accuracy')
        mlp.train(lr=mlp_setting['learning_rate'],
                  epochs=mlp_setting['train_epochs'])
        accu_test, accu_AIBL, accu_NACC, accu_FHS = mlp.test(i)[2:]
        accu['C']['test'].append(accu_test)
        accu['C']['NACC'].append(accu_NACC)
        accu['C']['AIBL'].append(accu_AIBL)
        accu['C']['FHS'].append(accu_FHS)


def mlp_D_train(exp_idx, repe_time, accu, config):
    # mlp build on CNN dense layer features and non-imaging features (age, gender, MMSE)  
    mlp_setting = config
    for i in range(repe_time):
        mlp = MLP_Wrapper_D(imbalan_ratio=mlp_setting['imbalan_ratio'],
                            fil_num=mlp_setting['fil_num'],
                            drop_rate=mlp_setting['drop_rate'],
                            batch_size=mlp_setting['batch_size'],
                            balanced=mlp_setting['balanced'],
                            exp_idx=exp_idx,
                            seed=seed + i,
                            model_name='mlp_D',
                            metric='accuracy')
        mlp.train(lr=mlp_setting['learning_rate'],
                  epochs=mlp_setting['train_epochs'])
        accu_test, accu_AIBL, accu_NACC, accu_FHS = mlp.test(i)[2:]
        accu['D']['test'].append(accu_test)
        accu['D']['NACC'].append(accu_NACC)
        accu['D']['AIBL'].append(accu_AIBL)
        accu['D']['FHS'].append(accu_FHS)


def mlp_E_train(exp_idx, repe_time, accu, config):
    # mlp build on age, gender, MMSE and apoe
    mlp_setting = config
    for i in range(repe_time):
        mlp = MLP_Wrapper_E(imbalan_ratio=mlp_setting['imbalan_ratio'],
                            fil_num=mlp_setting['fil_num'],
                            drop_rate=mlp_setting['drop_rate'],
                            batch_size=mlp_setting['batch_size'],
                            balanced=mlp_setting['balanced'],
                            roi_threshold=mlp_setting['roi_threshold'],
                            roi_count=mlp_setting['roi_count'],
                            choice=mlp_setting['choice'],
                            exp_idx=exp_idx,
                            seed=seed+i,
                            model_name='mlp_E',
                            metric='accuracy')
        mlp.train(lr=mlp_setting['learning_rate'],
                  epochs=mlp_setting['train_epochs'])
        accu_test, accu_AIBL, accu_NACC, accu_FHS = mlp.test(i)[2:]
        accu['E']['test'].append(accu_test)
        accu['E']['NACC'].append(accu_NACC)
        accu['E']['AIBL'].append(accu_AIBL)
        accu['E']['FHS'].append(accu_FHS)


def mlp_F_train(exp_idx, repe_time, accu, config):
    # mlp build on combined features of mlp_A and mlp_E
    mlp_setting = config
    for i in range(repe_time):
        mlp = MLP_Wrapper_F(imbalan_ratio=mlp_setting['imbalan_ratio'],
                            fil_num=mlp_setting['fil_num'],
                            drop_rate=mlp_setting['drop_rate'],
                            batch_size=mlp_setting['batch_size'],
                            balanced=mlp_setting['balanced'],
                            roi_threshold=mlp_setting['roi_threshold'],
                            roi_count=mlp_setting['roi_count'],
                            choice=mlp_setting['choice'],
                            exp_idx=exp_idx,
                            seed=seed+i,
                            model_name='mlp_F',
                            metric='accuracy')
        mlp.train(lr=mlp_setting['learning_rate'],
                  epochs=mlp_setting['train_epochs'])
        accu_test, accu_AIBL, accu_NACC, accu_FHS = mlp.test(i)[2:]
        accu['F']['test'].append(accu_test)
        accu['F']['NACC'].append(accu_NACC)
        accu['F']['AIBL'].append(accu_AIBL)
        accu['F']['FHS'].append(accu_FHS)


def mlp_A(config):
    print('##################################################')
    print(config)
    accu = {'A':{'test':[], 'NACC':[], 'AIBL':[], 'FHS':[]}}
    for exp_idx in range(repe_time):
        mlp_A_train(exp_idx, 3, accu, config)
    print('ADNI test accuracy ',
          'A {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['A']['test'])), float(np.std(accu['A']['test']))))
    print('NACC test accuracy ',
          'A {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['A']['NACC'])), float(np.std(accu['A']['NACC']))))
    print('AIBL test accuracy ',
          'A {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['A']['AIBL'])), float(np.std(accu['A']['AIBL']))))
    print('FHS test accuracy  ',
          'A {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['A']['FHS'])), float(np.std(accu['A']['FHS']))))


def mlp_B(config):
    print('##################################################')
    print(config)
    accu = {'B':{'test':[], 'NACC':[], 'AIBL':[], 'FHS':[]}}
    for exp_idx in range(repe_time):
        mlp_B_train(exp_idx, 3, accu, config)
    print('ADNI test accuracy ',
          'B {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['B']['test'])), float(np.std(accu['B']['test']))))
    print('NACC test accuracy ',
          'B {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['B']['NACC'])), float(np.std(accu['B']['NACC']))))
    print('AIBL test accuracy ',
          'B {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['B']['AIBL'])), float(np.std(accu['B']['AIBL']))))
    print('FHS test accuracy  ',
          'B {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['B']['FHS'])), float(np.std(accu['B']['FHS']))))


def mlp_C(config):
    print('##################################################')
    print(config)
    accu = {'C':{'test':[], 'NACC':[], 'AIBL':[], 'FHS':[]}}
    for exp_idx in range(repe_time):
        mlp_C_train(exp_idx, 3, accu, config)
    print('ADNI test accuracy ',
          'C {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['C']['test'])), float(np.std(accu['C']['test']))))
    print('NACC test accuracy ',
          'C {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['C']['NACC'])), float(np.std(accu['C']['NACC']))))
    print('AIBL test accuracy ',
          'C {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['C']['AIBL'])), float(np.std(accu['C']['AIBL']))))
    print('FHS test accuracy  ',
          'C {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['C']['FHS'])), float(np.std(accu['C']['FHS']))))


def mlp_D(config):
    print('##################################################')
    print(config)
    accu = {'D':{'test':[], 'NACC':[], 'AIBL':[], 'FHS':[]}}
    for exp_idx in range(repe_time):
        mlp_D_train(exp_idx, 3, accu, config)
    print('ADNI test accuracy ',
          'D {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['D']['test'])), float(np.std(accu['D']['test']))))
    print('NACC test accuracy ',
          'D {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['D']['NACC'])), float(np.std(accu['D']['NACC']))))
    print('AIBL test accuracy ',
          'D {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['D']['AIBL'])), float(np.std(accu['D']['AIBL']))))
    print('FHS test accuracy  ',
          'D {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['D']['FHS'])), float(np.std(accu['D']['FHS']))))


def mlp_E(config):
    print('##################################################')
    print(config)
    accu = {'E':{'test':[], 'NACC':[], 'AIBL':[], 'FHS':[]}}
    for exp_idx in range(repe_time):
        mlp_E_train(exp_idx, 3, accu, config)
    print('ADNI test accuracy ',
          'E {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['E']['test'])), float(np.std(accu['E']['test']))))
    print('NACC test accuracy ',
          'E {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['E']['NACC'])), float(np.std(accu['E']['NACC']))))
    print('AIBL test accuracy ',
          'E {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['E']['AIBL'])), float(np.std(accu['E']['AIBL']))))
    print('FHS test accuracy  ',
          'E {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['E']['FHS'])), float(np.std(accu['E']['FHS']))))


def mlp_F(config):
    print('##################################################')
    print(config)
    accu = {'F':{'test':[], 'NACC':[], 'AIBL':[], 'FHS':[]}}
    for exp_idx in range(repe_time):
        mlp_F_train(exp_idx, 3, accu, config)
    print('ADNI test accuracy ',
          'F {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['F']['test'])), float(np.std(accu['F']['test']))))
    print('NACC test accuracy ',
          'F {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['F']['NACC'])), float(np.std(accu['F']['NACC']))))
    print('AIBL test accuracy ',
          'F {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['F']['AIBL'])), float(np.std(accu['F']['AIBL']))))
    print('FHS test accuracy  ',
          'F {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['F']['FHS'])), float(np.std(accu['F']['FHS']))))


if __name__ == "__main__":
    config = read_json('./config.json')
    seed, repe_time = 1000, config['repeat_time']
    mlp_A(config["mlp_A"])
    mlp_B(config["mlp_B"])
    mlp_C(config["mlp_C"])
    mlp_D(config["mlp_D"])
    mlp_E(config["mlp_E"])
    mlp_F(config['mlp_F'])

    