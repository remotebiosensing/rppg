import torch
from torch.utils.data import Dataset
import numpy as np


class BPNetDataset(Dataset):
    def __init__(self, x_data, y_data, size_factor):
        self.x_data = torch.FloatTensor(x_data)
        # self.x_data = x_data
        self.y_data = torch.FloatTensor(y_data)
        # self.y_data = y_data
        self.size = torch.FloatTensor(size_factor)
        # self.size = size_factor
        self.len = self.y_data.shape[0]

    def __getitem__(self, index):
        x = self.x_data[index].to('cuda')
        y = self.y_data[index].to('cuda')
        '''
        size[index][0] = np.min(diastolic list) 
        size[index][1] = np.max(systolic list) 
        '''
        d = self.size[index][0].to('cuda')
        s = self.size[index][1].to('cuda')
        m = self.size[index][2].to('cuda')

        return x, y, d, s, m

    def __len__(self):
        return self.len
