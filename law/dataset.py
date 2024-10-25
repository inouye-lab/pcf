import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import sklearn.preprocessing as preprocessing
from pathlib import Path
import matplotlib.pyplot as plt

class SimLaw(data_utils.Dataset):
    def __init__(self,
                 split='train', 
                 root = '../VAE/saved/xxx'):
        
        self.split = split 
        self.root = root

        self._get_data()

    def _get_data(self):

        # this load the semi-synthetic dataset, the original law dataset we used is from https://github.com/osu-srml/cf_representation_learning
        # we used the same semi-synthetic dataset for different seed (seed=1), note that this does not mean the estimated VAE would be the same 
        data_dir = Path(f'{self.root}/law/gt/cvae/a_r_1.0_a_d_1.0_a_y_1.0_a_f_0.0_u_3_run_1_use_label_True')  / "data_dict.pth"

        data_dict = torch.load(data_dir)[self.split]

        self.r = data_dict['x'][:, :8]
        self.d = data_dict['x'][:, 8:]
        assert self.r.shape[1] == 8, 'check r dim'
        assert self.d.shape[1] == 2, 'check d dim'
        self.x = data_dict['x']
        #self.a = data_dict['a'].squeeze(1).to(torch.int64)
        self.a = data_dict['a'].to(torch.int64)
        self.y = data_dict['y']
        self.u = data_dict['u']
        self.x_cf = data_dict['x_cf']
        self.a_cf = data_dict['a_cf']
        self.y_cf = data_dict['y_cf']

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):

        r = self.r[index]
        d = self.d[index]
        a = self.a[index]
        y = self.y[index]
    
        return r, d, a, y
