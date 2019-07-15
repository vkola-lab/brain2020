# -*- coding: utf-8 -*-
"""
Created on Sat May 25 16:51:10 2019

@author: Iluva
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np


class _Dataset(Dataset):
    def __init__(self, X_bn, X_no_bn, y):
        self.X_bn = X_bn
        self.X_no_bn = X_no_bn
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        X_bn_sample = self.X_bn[idx] if self.X_bn is not None else None
        X_no_bn_sample = self.X_no_bn[idx] if self.X_no_bn is not None else None
        y_sample = self.y[idx]
        return X_bn_sample, X_no_bn_sample, y_sample
    
    
def _collate_fn(batch):
    X_bn_batch = np.stack([s[0] for s in batch]) if batch[0][0] is not None else None
    X_no_bn_batch = np.stack([s[1] for s in batch]) if batch[0][1] is not None else None
    y_batch = np.stack([s[2] for s in batch])
    return X_bn_batch, X_no_bn_batch, y_batch
        

class _NetMLP(nn.Module):
    def __init__(self, dr, hw, dim_bn, dim_no_bn):
        super(_NetMLP, self).__init__()        
        self.bn1 = nn.BatchNorm1d(dim_bn) if dim_bn != 0 else None
        self.fc1 = nn.Linear(dim_no_bn+dim_bn, hw)
        self.fc2 = nn.Linear(hw, 2)
        self.do1 = nn.Dropout(dr)
        self.do2 = nn.Dropout(dr) 
        self.ac1 = nn.LeakyReLU()
    
    def forward(self, X_bn, X_no_bn):
        if X_bn is None and X_no_bn is not None:
            X = X_no_bn
        elif X_bn is not None and X_no_bn is None:
            X = X_bn
        else:
            X = torch.cat((X_no_bn, self.bn1(X_bn)), 1)  
        out = self.do1(X)
        out = self.fc1(out)
        out = self.ac1(out)
        out = self.do2(out)
        out = self.fc2(out)
        return out

    
class NetMLP():
    def __init__(self, dim_bn, dim_no_bn, device='cpu', batch_size=10, learning_rate=.01, 
                 dropout_rate=.5, balance=2.3, hidden_width=64):
        self.net = _NetMLP(dropout_rate, hidden_width, dim_bn, dim_no_bn).to(device)
        self.crit = nn.CrossEntropyLoss(weight=torch.Tensor([1.9, balance])).to(device)
        self.device = device
        self.op = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.bs = batch_size
    
    def fit(self, X_bn, X_no_bn, y, n_epoch):
        self.net.train()
        data_loader = _Dataset(X_bn, X_no_bn, y)
        data_loader = DataLoader(data_loader, batch_size=self.bs, shuffle=True, drop_last=True, collate_fn=_collate_fn)
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
        
        