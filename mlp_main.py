from model_wraper import MLP_Wrapper_A, MLP_Wrapper_B, MLP_Wrapper_C
import sys
import torch
import json
from sklearn_classifiers import hypertune

def read_json(config_file):
    with open(config_file) as config_buffer:
        config = json.loads(config_buffer.read())
    return config

def mlp_main(config_file):
    mlp_setting = read_json(config_file)
    for exp_idx in range(1):
        mlp = MLP_Wrapper_C(fil_num         = mlp_setting['fil_num'],
                            drop_rate       = mlp_setting['drop_rate'],
                            batch_size      = mlp_setting['batch_size'],
                            balanced        = mlp_setting['balanced'],
                            roi_threshold   = mlp_setting['roi_threshold'],
                            exp_idx         = 1,
                            seed            = 1000,
                            model_name      = 'mlp_C',
                            metric          = 'accuracy')
        valid_optimal_accu = mlp.train(lr     = mlp_setting['learning_rate'],
                                       epochs = mlp_setting['train_epochs'])
    print('$' + str(1-valid_optimal_accu) + '$$')


if __name__ == '__main__':
    filename = sys.argv[1]
    with torch.cuda.device(3):
        hypertune(filename)


"""
MLPA
1 layer
Minimum expected objective value under model is 0.09638 (+/- 0.04546), at location:
                NAME          TYPE       VALUE
                ----          ----       -----
                train_epochs  int        292
                learning_rat  float      0.013587
                batch_size    int        77
                fil_num       int        175
                alpha         float      0.082764
                roi_threshol  float      0.332056
 
2 layer               
Minimum expected objective value under model is 0.11500 (+/- 0.00555), at location:
                NAME          TYPE       VALUE
                ----          ----       -----
                fil_num1      int        72          
                fil_num2      int        99          
                train_epochs  int        200         
                learning_rat  float      0.021655    
                batch_size    int        30          
                alpha         float      0.061462    
                roi_threshol  float      0.627131    

Minimum of observed values is 0.114943, at location:
                NAME          TYPE       VALUE
                ----          ----       -----
                fil_num1      int        200         
                fil_num2      int        100         
                train_epochs  int        440         
                learning_rat  float      0.000629    
                batch_size    int        128         
                alpha         float      0.378975    
                roi_threshol  float      0.000000

"""
