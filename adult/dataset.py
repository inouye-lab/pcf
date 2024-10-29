import torch
import torch.utils.data as data_utils
from pathlib import Path


class SimUCIAdult(data_utils.Dataset):
    def __init__(self,
                 split='train', 
                 root = '../VAE/saved/...'):
        
        self.split = split 
        self.root = root

        self._get_data()


    def _get_data(self):

        data_dir = Path(f'{self.root}/adult/gt/dcevae/a_r_1_a_d_1_a_y_1_a_h_0.1_a_f_0.0_u_0.5_ur_3_ud_4_run_1_use_label_True') / "data_dict.pth"

        data_dict = torch.load(data_dir)[self.split]

        self.r = data_dict['x'][:, :3]
        self.d = data_dict['x'][:, 3:]
        self.x = data_dict['x']
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
    


class SimUCIAdultTrain(data_utils.Dataset):
    def __init__(self,
                 split='train', 
                 root = '../VAE/saved/...'):
        
        self.split = split 
        self.root = root

        self._get_data()


    def _get_data(self):

        data_dir = Path(f'{self.root}/adult/gt/dcevae/a_r_1_a_d_1_a_y_1_a_h_0.1_a_f_0.0_u_0.5_ur_3_ud_4_run_1_use_label_True') / "data_dict.pth"

        data_dict = torch.load(data_dir)[self.split]

        self.r = data_dict['x'][:, :3]
        self.d = data_dict['x'][:, 3:]
        self.x = data_dict['x']
        #self.a = data_dict['a'].squeeze(1).to(torch.int64)
        self.a = data_dict['a'].to(torch.int64)
        self.y = data_dict['y']
        self.u = data_dict['u']
        self.x_cf = data_dict['x_cf']
        self.a_cf = data_dict['a_cf']
        self.y_cf = data_dict['y_cf']
        
        self.r2 = self.r.clone()
        shuffle = torch.randperm(self.r2.size(0))
        self.r2 = self.r2[shuffle]
        self.d2 = self.d.clone()
        self.d2 = self.d2[shuffle]
        self.a2 = self.a.clone()
        self.a2 = self.a2[shuffle]
        self.y2 = self.y.clone()
        self.y2 = self.y2[shuffle]
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):

        r = self.r[index]
        d = self.d[index]
        a = self.a[index]
        y = self.y[index]
    
        r2 = self.r2[index]
        d2 = self.d2[index]
        a2 = self.a2[index]
        y2 = self.y2[index]
    
        return r, d, a, y, r2, d2, a2, y2