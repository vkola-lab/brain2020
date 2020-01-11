import os
import numpy as np
from model import _CNN, _FCN, _MLP
from utils import matrix_sum, get_accu, get_MCC, get_confusion_matrix, write_raw_score, DPM_statistics, timeit, read_csv
from dataloader import CNN_Data, FCN_Data, MLP_Data, _MLP_collate_fn
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import numpy as np

"""
model wraper class are defined in this scripts which includes the following methods:
    1. init: initialize dataloader, model
    2. train:
    3. valid:
    4. test:
    5. ...

    1. FCN wraper

    2. MLP wraper

    3. CNN wraper

"""

class CNN_Wraper:
    def __init__(self, fil_num, drop_rate, seed, batch_size, balanced, Data_dir, exp_idx, model_name, metric):
        self.seed = seed
        self.exp_idx = exp_idx
        self.model_name = model_name
        self.eval_metric = get_accu if metric == 'accuracy' else get_MCC
        self.model = _CNN(num=fil_num, p=drop_rate).cuda()
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
        f = open(self.checkpoint_dir + 'raw_score_seed{}'.format(self.seed) + '.txt', 'w')
        self.model.load_state_dict(torch.load('{}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch)))
        with torch.no_grad():
            self.model.train(False)
            test_matrix = [[0, 0], [0, 0]]
            for inputs, labels in self.test_dataloader:
                inputs, labels = inputs.cuda(), labels.cuda()
                preds = self.model(inputs)
                write_raw_score(f, preds, labels)
                test_matrix = matrix_sum(test_matrix, get_confusion_matrix(preds, labels))
        print('Test confusion matrix:', test_matrix, 'test_metric:', "%.4f" % self.eval_metric(test_matrix))
        f.close()
        return self.eval_metric(test_matrix)

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


class FCN_Wraper(CNN_Wraper):
    def __init__(self, fil_num, drop_rate, seed, batch_size, balanced, Data_dir, exp_idx, model_name, metric, patch_size):
        self.seed = seed
        self.exp_idx = exp_idx
        self.patch_size = patch_size
        self.model_name = model_name
        self.eval_metric = get_accu if metric == 'accuracy' else get_MCC
        self.model = _FCN(num=fil_num, p=drop_rate).cuda()
        self.prepare_dataloader(batch_size, balanced, Data_dir)
        self.checkpoint_dir = './checkpoint_dir/{}_exp{}/'.format(self.model_name, exp_idx)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
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
            for stage in ['train', 'valid', 'test', 'AIBL', 'NACC']:
                if stage in ['AIBL', 'NACC', 'FHS']:
                    Data_dir = '/data/datasets/{}/'.format(stage)
                else:
                    Data_dir = '/data/datasets/ADNI_NoBack/'
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
                print(stage + ' confusion matrix ', matrix)
        print('DPM generation is done')

                
class MLP_Wrapper():
    def __init__(self, dim_bn, dim_no_bn, device='cpu', batch_size=10, learning_rate=.01, 
                 dropout_rate=.5, balance=2.3, hidden_width=64):
        self.net = _MLP(dropout_rate, hidden_width, dim_bn, dim_no_bn).to(device)
        self.crit = nn.CrossEntropyLoss(weight=torch.Tensor([1.9, balance])).to(device)
        self.device = device
        self.op = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.bs = batch_size
    
    def fit(self, X_bn, X_no_bn, y, n_epoch):
        self.net.train()
        data = MLP_Data(X_bn, X_no_bn, y)
        data_loader = DataLoader(data, batch_size=self.bs, shuffle=True, drop_last=True, collate_fn=_MLP_collate_fn)
        for epoch in tqdm(range(n_epoch)):
            cum_loss = 0
            for X_bn_batch, X_no_bn_batch, y_batch in data_loader:
                self.net.zero_grad()
                X_bn_batch = self._to_tensor(X_bn_batch, torch.float32)
                X_no_bn_batch = self._to_tensor(X_no_bn_batch, torch.float32)
                y_batch = self._to_tensor(y_batch, torch.long)
                out = self.net(X_bn_batch, X_no_bn_batch)
                loss = self.crit(out, y_batch)
                loss.backward()
                self.op.step()
                cum_loss += loss.detach().cpu()
            print('cum_loss: {}'.format(cum_loss))
                
    def eval(self, X_bn, X_no_bn):
        self.net.eval()
        with torch.no_grad():
            X_bn = self._to_tensor(X_bn, torch.float32)
            X_no_bn = self._to_tensor(X_no_bn, torch.float32)
            score = self.net(X_bn, X_no_bn).cpu().numpy()
        return score
    
    def _to_tensor(self, ndarray, dtype):
        tensor = torch.tensor(ndarray, dtype=dtype, device=self.device) if ndarray is not None else None
        return tensor


if __name__ == "__main__":
    pass
