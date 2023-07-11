from utils import read_json, data_split
from model_wrapper import CNN_Wrapper, FCN_Wrapper
import torch
torch.backends.cudnn.benchmark = True


def cnn_main(seed):
    cnn_setting = config['cnn']
    for exp_idx in range(repe_time):
        cnn = CNN_Wrapper(fil_num         = cnn_setting['fil_num'],
                          drop_rate       = cnn_setting['drop_rate'],
                          batch_size      = cnn_setting['batch_size'],
                          balanced        = cnn_setting['balanced'],
                          Data_dir        = cnn_setting['Data_dir'],
                          exp_idx         = exp_idx,
                          seed            = seed,
                          model_name      = 'cnn',
                          metric          = 'accuracy')
        cnn.train(lr     = cnn_setting['learning_rate'],
                  epochs = cnn_setting['train_epochs'])
        cnn.test()
        cnn.gen_features()


def fcn_main(seed):
    fcn_setting = config['fcn']
    for exp_idx in range(repe_time):
        fcn = FCN_Wrapper(fil_num        = fcn_setting['fil_num'],
                        drop_rate       = fcn_setting['drop_rate'],
                        batch_size      = fcn_setting['batch_size'],
                        balanced        = fcn_setting['balanced'],
                        Data_dir        = fcn_setting['Data_dir'],
                        patch_size      = fcn_setting['patch_size'],
                        exp_idx         = exp_idx,
                        seed            = seed,
                        model_name      = 'fcn',
                        metric          = 'accuracy')
        fcn.train(lr     = fcn_setting['learning_rate'],
                  epochs = fcn_setting['train_epochs'])
        fcn.test_and_generate_DPMs()


if __name__ == "__main__":

    config = read_json('./config.json')
    seed, repe_time = 1000, config['repeat_time']  # if you only want to use 1 data split, set repe_time = 1
    # data_split function splits ADNI dataset into training, validation and testing for several times (repe_time)
    data_split(repe_time=repe_time)

    # to perform FCN training #####################################
    with torch.cuda.device(1):  # specify which gpu to use
        fcn_main(seed)  # each FCN model will be independently trained on the corresponding data split

    # to perform CNN training #####################################
    # with torch.cuda.device(1): # specify which gpu to use
    #     cnn_main(seed)  # each CNN model will be independently trained on the corresponding data split
        



