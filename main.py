from utils import read_json, data_split
from model_wraper import CNN_Wraper, FCN_Wraper
import torch
torch.backends.cudnn.benchmark = True

seed = 1000
config = read_json('./config.json')
repe_time = config["repeat_time"]

if False: # if need to re-split the data
    data_split(repe_time=repe_time)

def cnn_main():
    # CNN training and validation
    cnn_setting = config['cnn']
    for exp_idx in range(repe_time):
        cnn = CNN_Wraper(fil_num        = cnn_setting['fil_num'], 
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
    
def fcn_main():
    # FCN training and validation
    fcn_setting = config['fcn']
    for exp_idx in range(repe_time):
        fcn = FCN_Wraper(fil_num        = fcn_setting['fil_num'],
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
    with torch.cuda.device(3):
        fcn_main()



