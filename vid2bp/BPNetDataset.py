import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt


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
        x = self.x_data[index][0].to('cuda')
        dx = torch.diff(x, dim=0)[89:269]
        # torch.cat((torch.diff(x), torch.zeros(1).to('cuda:0')))
        ddx = torch.diff(torch.diff(x, dim=0), dim=0)[88:268]
        # x = (self.x_data[index] - torch.mean(self.x_data[index])) / torch.std(self.x_data[index]).to('cuda')
        # x_size = torch.mean(self.x_data[index]).to('cuda')
        # if x_size < 0.5:
        #     size_class = 1
        # elif x_size < 1.0:
        #     size_class = 2
        # elif x_size < 1.5:
        #     size_class = 3
        # elif x_size < 2.0:
        #     size_class = 4
        # else:
        #     size_class = 5
        y = self.y_data[index].to('cuda')
        dy = torch.diff(y, dim=0)[89:269]
        ddy = torch.diff(torch.diff(y, dim=0), dim=0)[88:268]
        '''
        size[index][0] = np.min(diastolic list) 
        size[index][1] = np.max(systolic list) 
        '''
        d = self.size[index][0].to('cuda')
        s = self.size[index][1].to('cuda')
        # m = self.size[index][2].to('cuda')

        return x, dx, ddx, y, dy, ddy, d, s

    def __len__(self):
        return self.len
