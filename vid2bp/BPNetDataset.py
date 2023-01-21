import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt


class BPNetDataset(Dataset):
    def __init__(self, x_data, y_data, size_factor, device):
        self.x_data = torch.FloatTensor(x_data).to(device, non_blocking=True)
        # self.x_data = x_data
        self.y_data = torch.FloatTensor(y_data).to(device, non_blocking=True)
        # self.y_data = y_data
        self.size = torch.FloatTensor(size_factor).to(device, non_blocking=True)
        # self.size = size_factor
        self.len = self.y_data.shape[0]

    def __getitem__(self, index):
        x = self.x_data[index]
        # dx = torch.diff(x, dim=0)[89:269]
        # torch.cat((torch.diff(x), torch.zeros(1).to('cuda:0')))
        # ddx = torch.diff(torch.diff(x, dim=0), dim=0)[88:268]

        y = self.y_data[index]
        dy = torch.diff(y, dim=0)[89:269]
        ddy = torch.diff(torch.diff(y, dim=0), dim=0)[88:268]
        '''
        size[index][0] = np.min(diastolic list) 
        size[index][1] = np.max(systolic list) 
        '''
        d = self.size[index][0]
        s = self.size[index][1]
        # m = self.size[index][2].to('cuda')

        return x, y, dy, ddy, d, s

    def __len__(self):
        return self.len
