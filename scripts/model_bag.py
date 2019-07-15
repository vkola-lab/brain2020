#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 12:51:57 2019

@author: cxue2
"""

import torch
import os
import shutil
import numpy as np
from net_mlp import NetMLP

class ModelBag():
    def __init__(self, model_kwargs, n_model=None):
        if n_model:
            self.models = [NetMLP(**model_kwargs) for i in range(n_model)]
        else:
            self.models = []
        
    def fit(self, fit_kwargs):
        for model in self.models:
            model.fit(**fit_kwargs)
            
    def eval(self, eval_kwargs):
        all_scores = []
        for model in self.models:
            scores = model.eval(**eval_kwargs)
            all_scores.append(scores)
        return np.stack(all_scores)
    
    def save(self, path):
        # remove and remake if the folder exists 
        shutil.rmtree(path, ignore_errors=True)
        os.mkdir(path)
        for idx, model in enumerate(self.models):
            torch.save(model, '{}/mlp_{:0>3}'.format(path, idx))
            
    def load(self, path):
        self.models = []
        fs = os.listdir(path)
        for f in fs:
            model = torch.load('{}/{}'.format(path, f), map_location=lambda storage, location: storage)
            model.device = 'cpu'
            self.models.append(model)
            
        
        
    