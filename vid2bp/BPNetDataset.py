import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt


class BPNetDataset(Dataset):
    def __init__(self, x_data, y_data, size_factor, p_info, ohe, device):
        self.x_data = torch.FloatTensor(x_data).to(device, non_blocking=True)
        self.y_data = torch.FloatTensor(y_data).to(device, non_blocking=True)
        self.size = torch.FloatTensor(size_factor).to(device, non_blocking=True)
        # self.size = torch.IntTensor(size_factor).to(device, non_blocking=True)
        self.info = torch.FloatTensor(np.array(p_info, dtype=float)).to(device, non_blocking=True)
        self.ohe = torch.FloatTensor(ohe).to(device, non_blocking=True)
        self.len = self.y_data.shape[0]

    def __getitem__(self, index):
        x = self.x_data[index]
        y = self.y_data[index]
        '''
        size[index][0] = np.min(diastolic list) 
        size[index][1] = np.max(systolic list) 
        '''
        d = self.size[index][0]
        s = self.size[index][1]
        m = self.size[index][2]
        info = self.info[index]
        o = self.ohe[index]
        # m = self.size[index][2].to('cuda')

        return x, y, d, s, m, info, o

    def __len__(self):
        return self.len
