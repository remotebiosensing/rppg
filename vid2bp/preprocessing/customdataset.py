import torch
from torch.utils.data import Dataset
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, x_data, y_data, size_factor):
        self.x_data = torch.FloatTensor(x_data)
        # self.x_data = x_data
        self.y_data = torch.FloatTensor(y_data)
        # self.y_data = y_data
        self.size = torch.FloatTensor(size_factor)
        # self.size = size_factor
        self.len = self.y_data.shape[0]

    def __getitem__(self, index):
        # self.x_data = torch.reshape(self.x_data, [1, 1, len(self.x_data)])
        x = self.x_data[index].to('cuda')
        y = self.y_data[index].to('cuda')
        '''
        size[index][0] = np.min(diastolic list) 
        size[index][1] = np.max(systolic list) 
        '''
        d = self.size[index][0].to('cuda')
        s = self.size[index][1].to('cuda')

        return x, y, d, s

    def __len__(self):
        return self.len


class CustomDataset_Unet(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.FloatTensor(x_data)
        # print('shape of x_data : ', np.shape(self.x_data))
        self.y_data = torch.FloatTensor(y_data)
        # print('shape of y_data : ', np.shape(self.y_data))
        self.len = self.y_data.shape[0]

    def __getitem__(self, index):
        x = self.x_data[index].to('cuda', non_blocking=True)
        y = self.y_data[index].to('cuda', non_blocking=True)
        return x, y

    def __len__(self):
        return self.len
