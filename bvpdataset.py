import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import torchvision.transforms as transforms
import numpy as np



class dataset(Dataset):
    def __init__(self, A, M, labels ):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.a = A
        self.m = M
        self.labels = labels.reshape(-1, 1)

    def __getitem__(self, index, dtype=torch.float):
        if torch.is_tensor(index):
            index = index.tolist()

        norm_img = torch.tensor(np.transpose(self.a[index]), dtype=dtype)
        mot_img = torch.tensor(np.transpose(self.m[index]), dtype=dtype)
        # norm_img = torch.tensor(self.a[index], dtype=dtype)
        # mot_img = torch.tensor(self.m[index], dtype=dtype)
        labels = torch.tensor(self.labels[index], dtype=dtype)

        return norm_img, mot_img, labels

    def __len__(self):
        return len(self.labels)

class appdataset(Dataset):
    def __init__(self, A, labels ):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.a = A
        self.labels = labels.reshape(-1, 1)

    def __getitem__(self, index, dtype=torch.float):
        if torch.is_tensor(index):
            index = index.tolist()

        norm_img = torch.tensor(np.transpose(self.a[index]), dtype=dtype)
        labels = torch.tensor(self.labels[index], dtype=dtype)

        return norm_img,  labels

    def __len__(self):
        return len(self.labels)
class motdataset(Dataset):
    def __init__(self, M, labels ):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.M = M
        self.labels = labels.reshape(-1, 1)

    def __getitem__(self, index, dtype=torch.float):
        if torch.is_tensor(index):
            index = index.tolist()

        mot_img = torch.tensor(np.transpose(self.M[index]), dtype=dtype)
        labels = torch.tensor(self.labels[index], dtype=dtype)

        return mot_img,  labels

    def __len__(self):
        return len(self.labels)
