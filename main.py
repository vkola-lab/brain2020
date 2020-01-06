from utils import read_json, data_split
from model_wraper import CNN_Wraper, FCN_Wraper, MLP_Wraper

seed = 1000
config = read_json('./config.json')
print(config)

if False: # if need to re-split the data
    data_split(repe_time=5)

# CNN training and validation
cnn_setting = config['cnn']
for exp_idx in range(repe_time):
    cnn = CNN_Wraper(fil_num        = cnn_setting['fil_num'], 
                    drop_rate       = cnn_setting['drop_rate'], 
                    batch_size      = cnn_setting['batch_size'], 
                    balanced        = cnn_setting['balanced'], 
                    Data_dir        = cnn_setting['Data_dir'], 
                    exp_dir         = exp_idx,
                    seed            = seed)
    cnn.train(lr     = cnn_setting['learning_rate'],
              epochs = cnn_setting['train_epochs'])
    cnn.test()

# CNN training and validation
fcn_setting = config['fcn']
for exp_idx in range(repe_time):
    fcn = FCN_Wraper(fil_num        = fcn_setting['fil_num'], 
                    drop_rate       = fcn_setting['drop_rate'], 
                    batch_size      = fcn_setting['batch_size'], 
                    balanced        = fcn_setting['balanced'], 
                    Data_dir        = fcn_setting['Data_dir'], 
                    exp_dir         = exp_idx,
                    seed            = seed)
    fcn.train(lr     = fcn_setting['learning_rate'],
              epochs = fcn_setting['train_epochs'])
    fcn.test()


