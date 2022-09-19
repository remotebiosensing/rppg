import torch
from torch.utils.data import Dataset
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, x_data, y_data, size_factor):
        self.x_data = torch.FloatTensor(x_data)

        print('shape of x_data : ', np.shape(self.x_data))
        self.y_data = torch.FloatTensor(y_data)
        print('shape of y_data : ', np.shape(self.y_data))
        self.len = self.y_data.shape[0]
        self.size = torch.FloatTensor(size_factor)
        print('shape of size_data : ', np.shape(self.size))

    def __getitem__(self, index):
        # self.x_data = torch.reshape(self.x_data, [1, 1, len(self.x_data)])
        x = self.x_data[index].to('cuda')
        y = self.y_data[index].to('cuda')
        # d = torch.squeeze(self.size[index][0]).to('cuda')
        s = self.size[index][1].to('cuda')
        a = (self.size[index][1]-self.size[index][0]).to('cuda')
        '''variable a is dependent variable of s and d'''
        # m = self.size[index][2].to('cuda')
        return x, y, s, a #d, s, m

    def __len__(self):
        return self.len
