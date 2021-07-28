import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class DeepPhysDataset(Dataset):
    def __init__(self, appearance_data, motion_data, target):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.a = appearance_data
        self.m = motion_data
        self.label = target.reshape(-1, 1)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        appearance_data = torch.tensor(np.transpose(self.a[index], (2, 0, 1)), dtype=torch.float32)
        motion_data = torch.tensor(np.transpose(self.m[index], (2, 0, 1)), dtype=torch.float32)
        target = torch.tensor(self.label[index], dtype=torch.float32)

        inputs = torch.stack([appearance_data,motion_data],dim=0)

        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
            target = target.to('cuda')

        return inputs, target

    def __len__(self):
        return len(self.label)
