import os
from model import _CNN, _FCN, _CNN
from utils import matrix_sum, get_accu, get_confusion_matrix, write_raw_score
from dataloader import CNN_Data, FCN_Data, MLP_Data
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm

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
    def __init__(self, fil_num, drop_rate, seed, batch_size, balanced, Data_dir, exp_idx):
        self.seed = seed
        self.exp_idx = exp_idx
        self.model = _CNN(num=fil_num, p=drop_rate).cuda()
        self.prepare_dataloader(batch_size, balanced, Data_dir)
        self.checkpoint_dir = './checkpoint_dir/exp{}/'.format(exp_idx)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

    def train(self, lr, epochs):
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.999))
        self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, self.imbalanced_ratio])).cuda()
        self.optimal_valid_matrix = [[0, 0], [0, 0]]
        self.optimal_valid_accu   = 0
        self.optimal_epoch        = -1
        for self.epoch in range(epochs):
            self.train_model_epoch()
            valid_matrix = self.valid_model_epoch()
            print('This {}th epoch validation confusion matrix:'.format(self.epoch), valid_matrix, 'validation_accuracy:', "%.4f" % get_accu(valid_matrix))
            self.save_checkpoint(valid_matrix)
        print('(CNN) Best validation accuracy saved at the {}th epoch:'.format(self.optimal_epoch), self.optimal_valid_accu, self.optimal_valid_matrix)
        return self.valid_optimal_accu

    def test(self):
        f = open(self.checkpoint_dir + 'seed{}'.format(self.seed) + '.txt', 'w')
        self.model.load_state_dict(torch.load('{}CNN_{}.pth'.format(self.checkpoint_dir, self.optimal_epoch)))
        with torch.no_grad():
            self.model.train(False)
            test_matrix = [[0, 0], [0, 0]]
            for inputs, labels in self.test_dataloader:
                inputs, labels = inputs.cuda(), labels.cuda()
                preds = self.model(inputs)
                write_raw_score(f, preds, labels)
                test_matrix = matrix_sum(test_matrix, get_confusion_matrix(preds, labels))
        print('Test confusion matrix:', test_matrix, 'test_accuracy:', "%.4f" % get_accu(test_matrix))
        f.close()
        return get_accu(test_matrix)

    def save_checkpoint(self, valid_matrix):
        if get_accu(valid_matrix) >= self.optimal_valid_accu:
            self.optimal_epoch = self.epoch
            self.optimal_valid_matrix = valid_matrix
            self.optimal_valid_accu = get_accu(valid_matrix)
            for root, Dir, Files in os.walk(self.checkpoint_dir):
                for File in Files:
                    if File.endswith('.pth'):
                        try:
                            os.remove(self.checkpoint_dir + File)
                        except:
                            pass
            torch.save(self.model.state_dict(), '{}CNN_{}.pth'.format(self.checkpoint_dir, self.optimal_epoch))

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
        self.valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    pass
