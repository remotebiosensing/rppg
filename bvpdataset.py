import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import numpy as np
class bvpdataset(Dataset):
    def __init__(self,data_path,transform = None):
        self.transform = transform
        data = np.load(data_path)
        self.a = data['A']
        self.m = data['M']
        self.label = data['T'].reshape(-1,1)

    def __getitem__(self, index,dtype = torch.float):
        if torch.is_tensor(index):
            index = index.tolist()

        norm_img = torch.tensor(self.a[index],dtype = dtype)
        mot_img = torch.tensor(self.m[index],dtype = dtype)
        label = torch.tensor(self.label[index],dtype= dtype)


        return norm_img,mot_img,label

    def __len__(self):
        return len(self.label)