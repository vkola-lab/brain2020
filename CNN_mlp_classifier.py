from utils import read_json, data_split
from model_wraper import CNN_Wraper, FCN_Wraper, MLP_Wrapper_A, MLP_Wrapper_B, MLP_Wrapper_C
import torch
torch.backends.cudnn.benchmark = True

def gen_features():
    table = [27, 139, 46, 75, 112]
    cnn_setting = config['cnn']
    for exp_idx in range(5):
        cnn = CNN_Wraper(fil_num        = cnn_setting['fil_num'],
                        drop_rate       = cnn_setting['drop_rate'],
                        batch_size      = cnn_setting['batch_size'],
                        balanced        = cnn_setting['balanced'],
                        Data_dir        = cnn_setting['Data_dir'],
                        exp_idx         = exp_idx,
                        seed            = seed,
                        model_name      = 'cnn',
                        metric          = 'accuracy')
        cnn.optimal_epoch = table[exp_idx]
        cnn.gen_features()




if __name__ == "__main__":
    config = read_json('./config.json')
    seed = 1000
    gen_features()

