import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt


class BPNetDataset(Dataset):
    def __init__(self, ple, p_cycle, abp, a_cycle, dbp_label, sbp_label, p_info, device):
        self.x_data = torch.FloatTensor(ple).to(device, non_blocking=True)
        self.ple_cycle = torch.FloatTensor(p_cycle).to(device, non_blocking=True)
        self.y_data = torch.FloatTensor(abp).to(device, non_blocking=True)
        self.abp_cycle = torch.FloatTensor(a_cycle).to(device, non_blocking=True)
        self.dbp = torch.FloatTensor(dbp_label).to(device, non_blocking=True)
        self.sbp = torch.FloatTensor(sbp_label).to(device, non_blocking=True)
        # self.size = torch.FloatTensor(size_factor).to(device, non_blocking=True)
        self.info = torch.FloatTensor(np.array(p_info, dtype=float)).to(device, non_blocking=True)
        # self.ohe = torch.FloatTensor(ohe).to(device, non_blocking=True)
        self.len = self.y_data.shape[0]

    def __getitem__(self, index):
        x = self.x_data[index]
        x_cycle = self.ple_cycle[index]
        y = self.y_data[index]
        y_cycle = self.abp_cycle[index]
        '''
        size[index][0] = np.min(diastolic list) 
        size[index][1] = np.max(systolic list) 
        '''
        d = self.dbp[index][-1]
        s = self.sbp[index][-1]
        # d = self.size[index][0]
        # s = self.size[index][1]
        # m = self.size[index][2]
        info = self.info[index]
        # o = self.ohe[index]
        # m = self.size[index][2].to('cuda')

        return x, x_cycle, y, y_cycle, d, s, info  # returns 7 args

    def __len__(self):
        return self.len
