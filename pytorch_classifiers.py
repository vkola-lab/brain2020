from model_wraper import MLP_Wrapper_A, MLP_Wrapper_B, MLP_Wrapper_C
from utils import read_json
import numpy as np

seed = 1000
config = read_json('./config.json')

def mlp_A_train(exp_idx, repe_time, accu, config):
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

def mlp_B_train(exp_idx, repe_time, accu):
    mlp_setting = config['mlp_B']
    for i in range(repe_time):
        mlp = MLP_Wrapper_B(fil_num=mlp_setting['fil_num'],
                            drop_rate=mlp_setting['drop_rate'],
                            batch_size=mlp_setting['batch_size'],
                            balanced=mlp_setting['balanced'],
                            roi_threshold=mlp_setting['roi_threshold'],
                            exp_idx=exp_idx,
                            seed=seed+i,
                            model_name='mlp_B',
                            metric='accuracy')
        mlp.train(lr=mlp_setting['learning_rate'],
                  epochs=mlp_setting['train_epochs'])
        accu_test, accu_AIBL, accu_NACC, accu_FHS = mlp.test()[2:]
        accu['B']['test'].append(accu_test)
        accu['B']['NACC'].append(accu_NACC)
        accu['B']['AIBL'].append(accu_AIBL)
        accu['B']['FHS'].append(accu_FHS)

def mlp_C_train(exp_idx, repe_time, accu):
    mlp_setting = config['mlp_C']
    for i in range(repe_time):
        mlp = MLP_Wrapper_C(fil_num=mlp_setting['fil_num'],
                            drop_rate=mlp_setting['drop_rate'],
                            batch_size=mlp_setting['batch_size'],
                            balanced=mlp_setting['balanced'],
                            roi_threshold=mlp_setting['roi_threshold'],
                            exp_idx=exp_idx,
                            seed=seed+i,
                            model_name='mlp_C',
                            metric='accuracy')
        mlp.train(lr=mlp_setting['learning_rate'],
                  epochs=mlp_setting['train_epochs'])
        accu_test, accu_AIBL, accu_NACC, accu_FHS = mlp.test()[2:]
        accu['C']['test'].append(accu_test)
        accu['C']['NACC'].append(accu_NACC)
        accu['C']['AIBL'].append(accu_AIBL)
        accu['C']['FHS'].append(accu_FHS)

def mlp():
    accu = {'A':{'test':[], 'NACC':[], 'AIBL':[], 'FHS':[]}, \
            'B':{'test':[], 'NACC':[], 'AIBL':[], 'FHS':[]}, \
            'C':{'test':[], 'NACC':[], 'AIBL':[], 'FHS':[]}}

    for exp_idx in range(5):
        print('B')
        mlp_B_train(exp_idx, 3, accu)
        print('C')
        mlp_C_train(exp_idx, 3, accu)
        print('A')
        mlp_A_train(exp_idx, 3, accu)

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

def mlp_A(config):
    print(config)
    accu = {'A':{'test':[], 'NACC':[], 'AIBL':[], 'FHS':[]}}
    for exp_idx in range(5):
        print(exp_idx)
        mlp_A_train(exp_idx, 3, accu, config)
    print('##################################################')
    print(config)
    print('ADNI test accuracy ',
          'A {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['A']['test'])), float(np.std(accu['A']['test']))))
    print('NACC test accuracy ',
          'A {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['A']['NACC'])), float(np.std(accu['A']['NACC']))))
    print('AIBL test accuracy ',
          'A {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['A']['AIBL'])), float(np.std(accu['A']['AIBL']))))
    print('FHS test accuracy  ',
          'A {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['A']['FHS'])), float(np.std(accu['A']['FHS']))))

if __name__ == "__main__":
    config = read_json('./config.json')
    seed = 1000

    mlp_A(config["mlp_A"])

#     print(config['mlp_A'])
#     accu = {'A':{'test':[], 'NACC':[], 'AIBL':[], 'FHS':[]}}
#     mlp_A_train(2, 1, accu)
#     print(accu)
    