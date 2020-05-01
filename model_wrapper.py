import os
import numpy as np
from model import _CNN, _FCN, _MLP_A, _MLP_B, _MLP_C, _MLP_D
from utils import matrix_sum, get_accu, get_MCC, get_confusion_matrix, write_raw_score, DPM_statistics, timeit, read_csv
from dataloader import CNN_Data, FCN_Data, MLP_Data, MLP_Data_apoe, CNN_MLP_Data
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import numpy as np

"""
model wrapper class are defined in this scripts which includes the following methods:
    1. init: initialize dataloader, model
    2. train:
    3. valid:
    4. test:
    5. ...

    1. FCN wrapper

    2. MLP wrapper

    3. CNN wrapper

"""

class CNN_Wrapper:
    def __init__(self,
                 fil_num,
                 drop_rate,
                 seed,
                 batch_size,
                 balanced,
                 Data_dir,
                 exp_idx,
                 model_name,
                 metric):

        """
            :param fil_num:    output channel number of the first convolution layer
            :param drop_rate:  dropout rate of the last 2 layers, see model.py for details
            :param seed:       random seed
            :param batch_size: batch size for training CNN
            :param balanced:   balanced could take value 0 or 1, corresponding to different approaches to handle data imbalance,
                               see self.prepare_dataloader for more details
            :param Data_dir:   data path for training data
            :param exp_idx:    experiment index maps to different data splits
            :param model_name: give a name to the model
            :param metric:     metric used for saving model during training, can be either 'accuracy' or 'MCC'
                               for example, if metric == 'accuracy', then the time point where validation set has best accuracy will be saved
        """

        self.seed = seed
        self.exp_idx = exp_idx
        self.Data_dir = Data_dir
        self.model_name = model_name
        self.eval_metric = get_accu if metric == 'accuracy' else get_MCC
        self.model = _CNN(fil_num=fil_num, drop_rate=drop_rate).cuda()
        self.prepare_dataloader(batch_size, balanced, Data_dir)
        self.checkpoint_dir = './checkpoint_dir/{}_exp{}/'.format(self.model_name, exp_idx)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

    def train(self, lr, epochs):
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.999))
        self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, self.imbalanced_ratio])).cuda()
        self.optimal_valid_matrix = [[0, 0], [0, 0]]
        self.optimal_valid_metric = 0
        self.optimal_epoch        = -1
        for self.epoch in range(epochs):
            self.train_model_epoch()
            valid_matrix = self.valid_model_epoch()
            print('{}th epoch validation confusion matrix:'.format(self.epoch), valid_matrix, 'eval_metric:', "%.4f" % self.eval_metric(valid_matrix))
            self.save_checkpoint(valid_matrix)
        print('Best model saved at the {}th epoch:'.format(self.optimal_epoch), self.optimal_valid_metric, self.optimal_valid_matrix)
        return self.optimal_valid_metric

    def test(self):
        print('testing ... ')
        self.model.load_state_dict(torch.load('{}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch)))
        self.model.train(False)
        with torch.no_grad():
            for stage in ['train', 'valid', 'test', 'AIBL', 'NACC', 'FHS']:
                Data_dir = self.Data_dir
                if stage in ['AIBL', 'NACC', 'FHS']:
                    Data_dir = Data_dir.replace('ADNI', stage)
                data = CNN_Data(Data_dir, self.exp_idx, stage=stage, seed=self.seed)
                dataloader = DataLoader(data, batch_size=10, shuffle=False)
                f = open(self.checkpoint_dir + 'raw_score_{}.txt'.format(stage), 'w')
                matrix = [[0, 0], [0, 0]]
                for idx, (inputs, labels) in enumerate(dataloader):
                    inputs, labels = inputs.cuda(), labels.cuda()
                    preds = self.model(inputs)
                    write_raw_score(f, preds, labels)
                    matrix = matrix_sum(matrix, get_confusion_matrix(preds, labels))
                print(stage + ' confusion matrix ', matrix, ' accuracy ', self.eval_metric(matrix))
                f.close()

    def gen_features(self):
        self.model.load_state_dict(
            torch.load('{}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch)))
        self.model.train(False)
        self.feature_dir = './DPMs/{}_exp{}/'.format(self.model_name, self.exp_idx)
        if not os.path.exists(self.feature_dir):
            os.mkdir(self.feature_dir)
        with torch.no_grad():
            for stage in ['train', 'valid', 'test', 'AIBL', 'NACC', 'FHS']:
                Data_dir = self.Data_dir
                if stage in ['AIBL', 'NACC', 'FHS']:
                    Data_dir = Data_dir.replace('ADNI', stage)
                data = CNN_Data(Data_dir, self.exp_idx, stage=stage, seed=self.seed)
                filenames = data.Data_list
                dataloader = DataLoader(data, batch_size=1, shuffle=False)
                for idx, (inputs, labels) in enumerate(dataloader):
                    inputs, labels = inputs.cuda(), labels.cuda()
                    preds = self.model(inputs, stage="get_features").cpu().numpy().squeeze()
                    np.save(self.feature_dir + filenames[idx] + '.npy', preds)


    def save_checkpoint(self, valid_matrix):
        if self.eval_metric(valid_matrix) >= self.optimal_valid_metric:
            self.optimal_epoch = self.epoch
            self.optimal_valid_matrix = valid_matrix
            self.optimal_valid_metric = self.eval_metric(valid_matrix)
            for root, Dir, Files in os.walk(self.checkpoint_dir):
                for File in Files:
                    if File.endswith('.pth'):
                        try:
                            os.remove(self.checkpoint_dir + File)
                        except:
                            pass
            torch.save(self.model.state_dict(), '{}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))

    def train_model_epoch(self):
        self.model.train(True)
        for inputs, labels in self.train_dataloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            self.model.zero_grad()
            preds = self.model(inputs)
            loss = self.criterion(preds, labels)
            loss.backward()
            self.optimizer.step()

    def valid_model_epoch(self):
        with torch.no_grad():
            self.model.train(False)
            valid_matrix = [[0, 0], [0, 0]]
            for inputs, labels in self.valid_dataloader:
                inputs, labels = inputs.cuda(), labels.cuda()
                preds = self.model(inputs)
                valid_matrix = matrix_sum(valid_matrix, get_confusion_matrix(preds, labels))
        return valid_matrix

    def prepare_dataloader(self, batch_size, balanced, Data_dir):
        train_data = CNN_Data(Data_dir, self.exp_idx, stage='train', seed=self.seed)
        valid_data = CNN_Data(Data_dir, self.exp_idx, stage='valid', seed=self.seed)
        test_data  = CNN_Data(Data_dir, self.exp_idx, stage='test', seed=self.seed)
        sample_weight, self.imbalanced_ratio = train_data.get_sample_weights()
        # the following if else blocks represent two ways of handling class imbalance issue
        if balanced == 1:
            # use pytorch sampler to sample data with probability according to the count of each class
            # so that each mini-batch has the same expectation counts of samples from each class
            sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weight, len(sample_weight))
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
            self.imbalanced_ratio = 1
        elif balanced == 0:
            # sample data from the same probability, but
            # self.imbalanced_ratio will be used in the weighted cross entropy loss to handle imbalanced issue
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        self.valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
        self.test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


class FCN_Wrapper(CNN_Wrapper):

    def __init__(self, fil_num,
                       drop_rate,
                       seed,
                       batch_size,
                       balanced,
                       Data_dir,
                       exp_idx,
                       model_name,
                       metric,
                       patch_size):

        """
            :param fil_num:    output channel number of the first convolution layer
            :param drop_rate:  dropout rate of the last 2 layers, see model.py for details
            :param seed:       random seed
            :param batch_size: batch size for training FCN
            :param balanced:   balanced could take value 0 or 1, corresponding to different approaches to handle data imbalance,
                               see self.prepare_dataloader for more details
            :param Data_dir:   data path for training data
            :param exp_idx:    experiment index maps to different data splits
            :param model_name: give a name to the model
            :param metric:     metric used for saving model during training, can take 'accuracy' or 'MCC'
                               for example, if metric == 'accuracy', then the time point where validation set has best accuracy will be saved
            :param patch_size: size of patches for FCN training, must be 47. otherwise model has to be changed accordingly
        """

        self.seed = seed
        self.exp_idx = exp_idx
        self.Data_dir = Data_dir
        self.patch_size = patch_size
        self.model_name = model_name
        self.eval_metric = get_accu if metric == 'accuracy' else get_MCC
        self.model = _FCN(num=fil_num, p=drop_rate).cuda()
        self.prepare_dataloader(batch_size, balanced, Data_dir)
        if not os.path.exists('./checkpoint_dir/'): os.mkdir('./checkpoint_dir/')
        self.checkpoint_dir = './checkpoint_dir/{}_exp{}/'.format(self.model_name, exp_idx)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        if not os.path.exists('./DPMs/'): os.mkdir('./DPMs/')
        self.DPMs_dir = './DPMs/{}_exp{}/'.format(self.model_name, exp_idx)
        if not os.path.exists(self.DPMs_dir):
            os.mkdir(self.DPMs_dir)

    def train(self, lr, epochs):
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.999))
        self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, self.imbalanced_ratio])).cuda()
        self.optimal_valid_matrix = [[0, 0], [0, 0]]
        self.optimal_valid_metric = 0
        self.optimal_epoch        = -1
        for self.epoch in range(epochs):
            self.train_model_epoch()
            if self.epoch % 20 == 0:
                valid_matrix = self.valid_model_epoch()
                print('{}th epoch validation confusion matrix:'.format(self.epoch), valid_matrix, 'eval_metric:', "%.4f" % self.eval_metric(valid_matrix))
                self.save_checkpoint(valid_matrix)
        print('Best model saved at the {}th epoch:'.format(self.optimal_epoch), self.optimal_valid_metric, self.optimal_valid_matrix)
        return self.optimal_valid_metric

    def valid_model_epoch(self):
        self.fcn = self.model.dense_to_conv()
        DPMs, Labels = [], []
        with torch.no_grad():
            self.fcn.train(False)
            for idx, (inputs, labels) in enumerate(self.valid_dataloader):
                inputs, labels = inputs.cuda(), labels.cuda()
                DPM = self.fcn(inputs, stage='inference')
                DPMs.append(DPM.cpu().numpy().squeeze())
                Labels.append(labels)
        valid_matrix, ACCU, F1, MCC = DPM_statistics(DPMs, Labels)
        return valid_matrix

    def prepare_dataloader(self, batch_size, balanced, Data_dir):
        train_data = FCN_Data(Data_dir, self.exp_idx, stage='train', seed=self.seed, patch_size=self.patch_size)
        valid_data = FCN_Data(Data_dir, self.exp_idx, stage='valid', seed=self.seed, patch_size=self.patch_size)
        test_data  = FCN_Data(Data_dir, self.exp_idx, stage='test', seed=self.seed, patch_size=self.patch_size)
        sample_weight, self.imbalanced_ratio = train_data.get_sample_weights()
        # the following if else blocks represent two ways of handling class imbalance issue
        if balanced == 1:
            # use pytorch sampler to sample data with probability according to the count of each class
            # so that each mini-batch has the same expectation counts of samples from each class
            sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weight, len(sample_weight))
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
            self.imbalanced_ratio = 1
        elif balanced == 0:
            # sample data from the same probability, but
            # self.imbalanced_ratio will be used in the weighted cross entropy loss to handle imbalanced issue
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        self.valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False)
        self.test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

    def test_and_generate_DPMs(self):
        print('testing and generating DPMs ... ')
        self.model.load_state_dict(torch.load('{}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch)))
        self.fcn = self.model.dense_to_conv()
        self.fcn.train(False)
        with torch.no_grad():
            for stage in ['train', 'valid', 'test', 'AIBL', 'NACC', 'FHS']:
                Data_dir = self.Data_dir
                if stage in ['AIBL', 'NACC', 'FHS']:
                    Data_dir = Data_dir.replace('ADNI', stage)
                data = FCN_Data(Data_dir, self.exp_idx, stage=stage, whole_volume=True, seed=self.seed, patch_size=self.patch_size)
                filenames = data.Data_list
                dataloader = DataLoader(data, batch_size=1, shuffle=False)
                DPMs, Labels = [], []
                for idx, (inputs, labels) in enumerate(dataloader):
                    inputs, labels = inputs.cuda(), labels.cuda()
                    DPM = self.fcn(inputs, stage='inference').cpu().numpy().squeeze()
                    np.save(self.DPMs_dir + filenames[idx] + '.npy', DPM)
                    DPMs.append(DPM)
                    Labels.append(labels)
                matrix, ACCU, F1, MCC = DPM_statistics(DPMs, Labels) 
                np.save(self.DPMs_dir + '{}_MCC.npy'.format(stage), MCC)
                np.save(self.DPMs_dir + '{}_F1.npy'.format(stage),  F1)
                np.save(self.DPMs_dir + '{}_ACCU.npy'.format(stage), ACCU)  
                print(stage + ' confusion matrix ', matrix, ' accuracy ', self.eval_metric(matrix))
        print('DPM generation is done')


class MLP_Wrapper_A(CNN_Wrapper):
    def __init__(self, imbalan_ratio, fil_num, drop_rate, seed, batch_size, balanced, exp_idx, model_name, metric, roi_threshold, roi_count=200, choice='count'):
        self.seed = seed
        self.imbalan_ratio = imbalan_ratio
        self.choice = choice
        self.exp_idx = exp_idx
        self.model_name = model_name
        self.roi_count = roi_count
        self.roi_threshold = roi_threshold
        self.eval_metric = get_accu if metric == 'accuracy' else get_MCC
        self.checkpoint_dir = './checkpoint_dir/{}_exp{}/'.format(self.model_name, exp_idx)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.Data_dir = './DPMs/fcn_exp{}/'.format(exp_idx)
        self.prepare_dataloader(batch_size, balanced, self.Data_dir)
        self.model = _MLP_A(in_size=self.in_size, fil_num=fil_num, drop_rate=drop_rate)

    def prepare_dataloader(self, batch_size, balanced, Data_dir):
        train_data = MLP_Data(Data_dir, self.exp_idx, stage='train', roi_threshold=self.roi_threshold, roi_count=self.roi_count, choice=self.choice, seed=self.seed)
        valid_data = MLP_Data(Data_dir, self.exp_idx, stage='valid', roi_threshold=self.roi_threshold, roi_count=self.roi_count, choice=self.choice, seed=self.seed)
        test_data  = MLP_Data(Data_dir, self.exp_idx, stage='test', roi_threshold=self.roi_threshold, roi_count=self.roi_count, choice=self.choice, seed=self.seed)
        sample_weight, self.imbalanced_ratio = train_data.get_sample_weights()
        # the following if else blocks represent two ways of handling class imbalance issue
        if balanced == 1:
            # use pytorch sampler to sample data with probability according to the count of each class
            # so that each mini-batch has the same expectation counts of samples from each class
            sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weight, len(sample_weight))
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
            self.imbalanced_ratio = 1
        elif balanced == 0:
            # sample data from the same probability, but
            # self.imbalanced_ratio will be used in the weighted cross entropy loss to handle imbalanced issue
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
            self.imbalanced_ratio *= self.imbalan_ratio
        self.valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False)
        self.test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
        self.in_size = train_data.in_size
    
    def train(self, lr, epochs):
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.999))
        self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, self.imbalanced_ratio]))
        self.optimal_valid_matrix = [[0, 0], [0, 0]]
        self.optimal_valid_metric = 0
        self.optimal_epoch        = -1
        for self.epoch in range(epochs):
            self.train_model_epoch()
            valid_matrix = self.valid_model_epoch()
            #print('{}th epoch validation confusion matrix:'.format(self.epoch), valid_matrix, 'eval_metric:', "%.4f" % self.eval_metric(valid_matrix))
            self.save_checkpoint(valid_matrix)
        #print('Best model saved at the {}th epoch:'.format(self.optimal_epoch), self.optimal_valid_metric, self.optimal_valid_matrix)
        return self.optimal_valid_metric

    def train_model_epoch(self):
        self.model.train(True)
        for inputs, labels, _ in self.train_dataloader:
            inputs, labels = inputs, labels
            self.model.zero_grad()
            preds = self.model(inputs)
            loss = self.criterion(preds, labels)
            loss.backward()
            self.optimizer.step()

    def valid_model_epoch(self):
        with torch.no_grad():
            self.model.train(False)
            valid_matrix = [[0, 0], [0, 0]]
            for inputs, labels, _ in self.valid_dataloader:
                inputs, labels = inputs, labels
                preds = self.model(inputs)
                valid_matrix = matrix_sum(valid_matrix, get_confusion_matrix(preds, labels))
        return valid_matrix

    def test(self, repe_idx):
        self.model.load_state_dict(torch.load('{}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch)))
        self.model.train(False)
        accu_list = []
        with torch.no_grad():
            for stage in ['train', 'valid', 'test', 'AIBL', 'NACC', 'FHS_Full']:
                data = MLP_Data(self.Data_dir, self.exp_idx, stage=stage, roi_threshold=self.roi_threshold, roi_count=self.roi_count, choice=self.choice, seed=self.seed)
                dataloader = DataLoader(data, batch_size=10, shuffle=False)
                f = open(self.checkpoint_dir + 'raw_score_{}_{}.txt'.format(stage, repe_idx), 'w')
                matrix = [[0, 0], [0, 0]]
                for idx, (inputs, labels, _) in enumerate(dataloader):
                    inputs, labels = inputs, labels
                    preds = self.model(inputs)
                    write_raw_score(f, preds, labels)
                    matrix = matrix_sum(matrix, get_confusion_matrix(preds, labels))
                # print(stage + ' confusion matrix ', matrix, ' accuracy ', self.eval_metric(matrix))
                f.close()
                accu_list.append(self.eval_metric(matrix))
        return accu_list


class MLP_Wrapper_B(MLP_Wrapper_A):
    def __init__(self, imbalan_ratio, fil_num, drop_rate, seed, batch_size, balanced, exp_idx, model_name, metric, roi_threshold, roi_count, choice):
        super().__init__(imbalan_ratio, fil_num, drop_rate, seed, batch_size, balanced, exp_idx, model_name, metric, roi_threshold, roi_count, choice)
        self.model = _MLP_B(in_size=4, fil_num=fil_num, drop_rate=drop_rate)
    
    def train_model_epoch(self):
        self.model.train(True)
        for _, labels, inputs in self.train_dataloader:
            inputs, labels = inputs, labels
            self.model.zero_grad()
            preds = self.model(inputs)
            loss = self.criterion(preds, labels)
            loss.backward()
            self.optimizer.step()

    def valid_model_epoch(self):
        with torch.no_grad():
            self.model.train(False)
            valid_matrix = [[0, 0], [0, 0]]
            for _, labels, inputs in self.valid_dataloader:
                inputs, labels = inputs, labels
                preds = self.model(inputs)
                valid_matrix = matrix_sum(valid_matrix, get_confusion_matrix(preds, labels))
        return valid_matrix

    def test(self, repe_idx):
        accu_list = []
        self.model.load_state_dict(torch.load('{}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch)))
        self.model.train(False)
        with torch.no_grad():
            for stage in ['train', 'valid', 'test', 'AIBL', 'NACC', 'FHS']:
                data = MLP_Data(self.Data_dir, self.exp_idx, stage=stage, roi_threshold=self.roi_threshold, roi_count=self.roi_count, choice=self.choice, seed=self.seed)
                dataloader = DataLoader(data, batch_size=10, shuffle=False)
                f = open(self.checkpoint_dir + 'raw_score_{}_{}.txt'.format(stage, repe_idx), 'w')
                matrix = [[0, 0], [0, 0]]
                for idx, (_, labels, inputs) in enumerate(dataloader):
                    inputs, labels = inputs, labels
                    preds = self.model(inputs)
                    write_raw_score(f, preds, labels)
                    matrix = matrix_sum(matrix, get_confusion_matrix(preds, labels))
                # print(stage + ' confusion matrix ', matrix, ' accuracy ', self.eval_metric(matrix))
                f.close()
                accu_list.append(self.eval_metric(matrix))
        return accu_list


class MLP_Wrapper_C(MLP_Wrapper_A):
    def __init__(self, imbalan_ratio, fil_num, drop_rate, seed, batch_size, balanced, exp_idx, model_name, metric, roi_threshold, roi_count, choice):
        super().__init__(imbalan_ratio, fil_num, drop_rate, seed, batch_size, balanced, exp_idx, model_name, metric, roi_threshold, roi_count, choice)
        self.model = _MLP_C(in_size=self.in_size+4, fil_num=fil_num, drop_rate=drop_rate)
    
    def train_model_epoch(self):
        self.model.train(True)
        for inputs, labels, demors in self.train_dataloader:
            inputs, labels, demors = inputs, labels, demors
            self.model.zero_grad()
            preds = self.model(inputs, demors)
            loss = self.criterion(preds, labels)
            loss.backward()
            self.optimizer.step()

    def valid_model_epoch(self):
        with torch.no_grad():
            self.model.train(False)
            valid_matrix = [[0, 0], [0, 0]]
            for inputs, labels, demors in self.valid_dataloader:
                inputs, labels, demors = inputs, labels, demors
                preds = self.model(inputs, demors)
                valid_matrix = matrix_sum(valid_matrix, get_confusion_matrix(preds, labels))
        return valid_matrix

    def test(self, repe_idx):
        accu_list = []
        self.model.load_state_dict(torch.load('{}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch)))
        self.model.train(False)
        with torch.no_grad():
            for stage in ['train', 'valid', 'test', 'AIBL', 'NACC', 'FHS']:
                data = MLP_Data(self.Data_dir, self.exp_idx, stage=stage, roi_threshold=self.roi_threshold, roi_count=self.roi_count, choice=self.choice, seed=self.seed)
                dataloader = DataLoader(data, batch_size=10, shuffle=False)
                f = open(self.checkpoint_dir + 'raw_score_{}_{}.txt'.format(stage, repe_idx), 'w')
                matrix = [[0, 0], [0, 0]]
                for idx, (inputs, labels, demors) in enumerate(dataloader):
                    inputs, labels, demors = inputs, labels, demors
                    preds = self.model(inputs, demors)
                    write_raw_score(f, preds, labels)
                    matrix = matrix_sum(matrix, get_confusion_matrix(preds, labels))
                # print(stage + ' confusion matrix ', matrix, ' accuracy ', self.eval_metric(matrix))
                f.close()
                accu_list.append(self.eval_metric(matrix))
        return accu_list


class MLP_Wrapper_D(CNN_Wrapper):
    def __init__(self, imbalan_ratio, fil_num, drop_rate, seed, batch_size, balanced, exp_idx, model_name, metric):
        self.seed = seed
        self.imbalan_ratio = imbalan_ratio
        self.exp_idx = exp_idx
        self.model_name = model_name
        self.eval_metric = get_accu if metric == 'accuracy' else get_MCC
        self.checkpoint_dir = './checkpoint_dir/{}_exp{}/'.format(self.model_name, exp_idx)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.Data_dir = './DPMs/cnn_exp{}/'.format(exp_idx)
        self.prepare_dataloader(batch_size, balanced, self.Data_dir)
        self.model = _MLP_D(in_size=self.in_size+4, fil_num=fil_num, drop_rate=drop_rate)

    def prepare_dataloader(self, batch_size, balanced, Data_dir):
        train_data = CNN_MLP_Data(Data_dir, self.exp_idx, stage='train', seed=self.seed)
        valid_data = CNN_MLP_Data(Data_dir, self.exp_idx, stage='valid', seed=self.seed)
        test_data = CNN_MLP_Data(Data_dir, self.exp_idx, stage='test', seed=self.seed)
        sample_weight, self.imbalanced_ratio = train_data.get_sample_weights()
        # the following if else blocks represent two ways of handling class imbalance issue
        if balanced == 1:
            # use pytorch sampler to sample data with probability according to the count of each class
            # so that each mini-batch has the same expectation counts of samples from each class
            sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weight, len(sample_weight))
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
            self.imbalanced_ratio = 1
        elif balanced == 0:
            # sample data from the same probability, but
            # self.imbalanced_ratio will be used in the weighted cross entropy loss to handle imbalanced issue
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
            self.imbalanced_ratio *= self.imbalan_ratio
        self.valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False)
        self.test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
        self.in_size = train_data.in_size

    def train(self, lr, epochs):
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.999))
        self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, self.imbalanced_ratio]))
        self.optimal_valid_matrix = [[0, 0], [0, 0]]
        self.optimal_valid_metric = 0
        self.optimal_epoch = -1
        for self.epoch in range(epochs):
            self.train_model_epoch()
            valid_matrix = self.valid_model_epoch()
            # print('{}th epoch validation confusion matrix:'.format(self.epoch), valid_matrix, 'eval_metric:', "%.4f" % self.eval_metric(valid_matrix))
            self.save_checkpoint(valid_matrix)
        # print('Best model saved at the {}th epoch:'.format(self.optimal_epoch), self.optimal_valid_metric, self.optimal_valid_matrix)
        return self.optimal_valid_metric

    def train_model_epoch(self):
        self.model.train(True)
        for inputs, labels, demors in self.train_dataloader:
            inputs, labels = inputs, labels
            self.model.zero_grad()
            preds = self.model(inputs, demors)
            loss = self.criterion(preds, labels)
            loss.backward()
            self.optimizer.step()

    def valid_model_epoch(self):
        with torch.no_grad():
            self.model.train(False)
            valid_matrix = [[0, 0], [0, 0]]
            for inputs, labels, demors in self.valid_dataloader:
                inputs, labels = inputs, labels
                preds = self.model(inputs, demors)
                valid_matrix = matrix_sum(valid_matrix, get_confusion_matrix(preds, labels))
        return valid_matrix

    def test(self, repe_idx):
        self.model.load_state_dict(
            torch.load('{}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch)))
        self.model.train(False)
        accu_list = []
        with torch.no_grad():
            for stage in ['train', 'valid', 'test', 'AIBL', 'NACC', 'FHS']:
                data = CNN_MLP_Data(self.Data_dir, self.exp_idx, stage=stage, seed=self.seed)
                dataloader = DataLoader(data, batch_size=10, shuffle=False)
                f = open(self.checkpoint_dir + 'raw_score_{}_{}.txt'.format(stage, repe_idx), 'w')
                matrix = [[0, 0], [0, 0]]
                for idx, (inputs, labels, demors) in enumerate(dataloader):
                    inputs, labels = inputs, labels
                    preds = self.model(inputs, demors)
                    write_raw_score(f, preds, labels)
                    matrix = matrix_sum(matrix, get_confusion_matrix(preds, labels))
                print(stage + ' confusion matrix ', matrix, ' accuracy ', self.eval_metric(matrix))
                f.close()
                accu_list.append(self.eval_metric(matrix))
        return accu_list


class MLP_Wrapper_E(MLP_Wrapper_B):
    def __init__(self, imbalan_ratio, fil_num, drop_rate, seed, batch_size, balanced, exp_idx, model_name, metric, roi_threshold, roi_count, choice):
        super().__init__(imbalan_ratio, fil_num, drop_rate, seed, batch_size, balanced, exp_idx, model_name, metric, roi_threshold, roi_count, choice)
        self.prepare_dataloader(batch_size, balanced, self.Data_dir)
        self.model = _MLP_B(in_size=5, fil_num=fil_num, drop_rate=drop_rate)

    def prepare_dataloader(self, batch_size, balanced, Data_dir):
        train_data = MLP_Data_apoe(Data_dir, self.exp_idx, stage='train', roi_threshold=self.roi_threshold, roi_count=self.roi_count, choice=self.choice, seed=self.seed)
        valid_data = MLP_Data_apoe(Data_dir, self.exp_idx, stage='valid', roi_threshold=self.roi_threshold, roi_count=self.roi_count, choice=self.choice, seed=self.seed)
        test_data  = MLP_Data_apoe(Data_dir, self.exp_idx, stage='test', roi_threshold=self.roi_threshold, roi_count=self.roi_count, choice=self.choice, seed=self.seed)
        sample_weight, self.imbalanced_ratio = train_data.get_sample_weights()
        # the following if else blocks represent two ways of handling class imbalance issue
        if balanced == 1:
            # use pytorch sampler to sample data with probability according to the count of each class
            # so that each mini-batch has the same expectation counts of samples from each class
            sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weight, len(sample_weight))
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
            self.imbalanced_ratio = 1
        elif balanced == 0:
            # sample data from the same probability, but
            # self.imbalanced_ratio will be used in the weighted cross entropy loss to handle imbalanced issue
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
            self.imbalanced_ratio *= self.imbalan_ratio
        self.valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False)
        self.test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
        self.in_size = train_data.in_size

    def test(self, repe_idx):
        accu_list = []
        self.model.load_state_dict(torch.load('{}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch)))
        self.model.train(False)
        with torch.no_grad():
            for stage in ['train', 'valid', 'test', 'AIBL', 'NACC', 'FHS']:
                data = MLP_Data_apoe(self.Data_dir, self.exp_idx, stage=stage, roi_threshold=self.roi_threshold, roi_count=self.roi_count, choice=self.choice, seed=self.seed)
                dataloader = DataLoader(data, batch_size=10, shuffle=False)
                f = open(self.checkpoint_dir + 'raw_score_{}_{}.txt'.format(stage, repe_idx), 'w')
                matrix = [[0, 0], [0, 0]]
                for idx, (_, labels, inputs) in enumerate(dataloader):
                    inputs, labels = inputs, labels
                    preds = self.model(inputs)
                    write_raw_score(f, preds, labels)
                    matrix = matrix_sum(matrix, get_confusion_matrix(preds, labels))
                # print(stage + ' confusion matrix ', matrix, ' accuracy ', self.eval_metric(matrix))
                f.close()
                accu_list.append(self.eval_metric(matrix))
        return accu_list


class MLP_Wrapper_F(MLP_Wrapper_C):
    def __init__(self, imbalan_ratio, fil_num, drop_rate, seed, batch_size, balanced, exp_idx, model_name, metric, roi_threshold, roi_count, choice):
        super().__init__(imbalan_ratio, fil_num, drop_rate, seed, batch_size, balanced, exp_idx, model_name, metric, roi_threshold, roi_count, choice)
        self.prepare_dataloader(batch_size, balanced, self.Data_dir)
        self.model = _MLP_C(in_size=self.in_size + 5, fil_num=fil_num, drop_rate=drop_rate)

    def prepare_dataloader(self, batch_size, balanced, Data_dir):
        train_data = MLP_Data_apoe(Data_dir, self.exp_idx, stage='train', roi_threshold=self.roi_threshold, roi_count=self.roi_count, choice=self.choice, seed=self.seed)
        valid_data = MLP_Data_apoe(Data_dir, self.exp_idx, stage='valid', roi_threshold=self.roi_threshold, roi_count=self.roi_count, choice=self.choice, seed=self.seed)
        test_data  = MLP_Data_apoe(Data_dir, self.exp_idx, stage='test', roi_threshold=self.roi_threshold, roi_count=self.roi_count, choice=self.choice, seed=self.seed)
        sample_weight, self.imbalanced_ratio = train_data.get_sample_weights()
        # the following if else blocks represent two ways of handling class imbalance issue
        if balanced == 1:
            # use pytorch sampler to sample data with probability according to the count of each class
            # so that each mini-batch has the same expectation counts of samples from each class
            sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weight, len(sample_weight))
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
            self.imbalanced_ratio = 1
        elif balanced == 0:
            # sample data from the same probability, but
            # self.imbalanced_ratio will be used in the weighted cross entropy loss to handle imbalanced issue
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
            self.imbalanced_ratio *= self.imbalan_ratio
        self.valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False)
        self.test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
        self.in_size = train_data.in_size

    def test(self, repe_idx):
        accu_list = []
        self.model.load_state_dict(torch.load('{}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch)))
        self.model.train(False)
        with torch.no_grad():
            for stage in ['train', 'valid', 'test', 'AIBL', 'NACC', 'FHS']:
                data = MLP_Data_apoe(self.Data_dir, self.exp_idx, stage=stage, roi_threshold=self.roi_threshold, roi_count=self.roi_count, choice=self.choice, seed=self.seed)
                dataloader = DataLoader(data, batch_size=10, shuffle=False)
                f = open(self.checkpoint_dir + 'raw_score_{}_{}.txt'.format(stage, repe_idx), 'w')
                matrix = [[0, 0], [0, 0]]
                for idx, (inputs, labels, demors) in enumerate(dataloader):
                    inputs, labels, demors = inputs, labels, demors
                    preds = self.model(inputs, demors)
                    write_raw_score(f, preds, labels)
                    matrix = matrix_sum(matrix, get_confusion_matrix(preds, labels))
                # print(stage + ' confusion matrix ', matrix, ' accuracy ', self.eval_metric(matrix))
                f.close()
                accu_list.append(self.eval_metric(matrix))
        return accu_list



if __name__ == "__main__":
    pass
