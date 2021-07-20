import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np


class dataset(Dataset):
    def __init__(self, data, label):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.data = data
        self.label = label

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        data = torch.tensor(np.transpose(self.data[index], (3, 0, 1, 2)), dtype=torch.float32)
        label = torch.tensor(self.label[index], dtype=torch.float32)

        return data, label

    def __len__(self):
        return len(self.label)