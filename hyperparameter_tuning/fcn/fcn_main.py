from model_wraper import FCN_Wraper
import sys
import torch
import json

def read_json(config_file):
    with open(config_file) as config_buffer:
        config = json.loads(config_buffer.read())
    return config

def main(filename):
    AUTO = True
    for seed in range(1):
        print('training model with seed {}'.format(seed))
        cnn = CNN('./'+filename, seed)
        valid_optimal_accu = cnn.train()

    # ROC_plot(cnn.get_checkpoint_dir())

    if AUTO:
        print('$' + str(1-valid_optimal_accu) + '$$')

def fcn_main(config_file):
    fcn_setting = read_json(config_file)
    for exp_idx in range(1):
        fcn = FCN_Wraper(fil_num        = fcn_setting['fil_num'],
                        drop_rate       = fcn_setting['drop_rate'],
                        batch_size      = fcn_setting['batch_size'],
                        balanced        = fcn_setting['balanced'],
                        Data_dir        = fcn_setting['Data_dir'],
                        patch_size      = fcn_setting['patch_size'],
                        exp_idx         = exp_idx,
                        seed            = 1000,
                        model_name      = 'fcn',
                        metric          = 'accuracy')
        valid_optimal_accu = fcn.train(lr     = fcn_setting['learning_rate'],
                                       epochs = fcn_setting['train_epochs'])
    print('$' + str(1-valid_optimal_accu) + '$$')


if __name__ == '__main__':
    filename = sys.argv[1]
    #with torch.cuda.device(0):
    with torch.cuda.device(3):
        fcn_main(filename)
