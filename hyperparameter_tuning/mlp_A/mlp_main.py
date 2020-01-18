from model_wraper import MLP_Wrapper_A
import sys
import torch
import json

def read_json(config_file):
    with open(config_file) as config_buffer:
        config = json.loads(config_buffer.read())
    return config

def mlp_main(config_file):
    mlp_setting = read_json(config_file)
    for exp_idx in range(1):
        mlp = MLP_Wrapper_A(fil_num         = mlp_setting['fil_num'],
                            drop_rate       = mlp_setting['drop_rate'],
                            batch_size      = mlp_setting['batch_size'],
                            balanced        = mlp_setting['balanced'],
                            roi_threshold   = mlp_setting['roi_threshold'],
                            exp_idx         = 1,
                            seed            = 1000,
                            model_name      = 'mlp_A',
                            metric          = 'accuracy')
        valid_optimal_accu = mlp.train(lr     = mlp_setting['learning_rate'],
                                       epochs = mlp_setting['train_epochs'])
    print('$' + str(1-valid_optimal_accu) + '$$')


if __name__ == '__main__':
    filename = sys.argv[1]
    with torch.cuda.device(3):
        mlp_main(filename)
