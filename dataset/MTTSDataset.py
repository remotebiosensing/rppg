import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class MTTSDataset(Dataset):
    def __init__(self, appearance_data, motion_data, hr_target,rr_target):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.a = appearance_data
        self.m = motion_data
        self.hr_label = hr_target.reshape(-1, 1)
        self.rr_label = rr_target.reshape(-1, 1)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        appearance_data = torch.tensor(np.transpose(self.a[index], (2, 0, 1)), dtype=torch.float32)
        motion_data = torch.tensor(np.transpose(self.m[index], (2, 0, 1)), dtype=torch.float32)
        hr_target = torch.tensor(self.hr_label[index], dtype=torch.float32)
        rr_target = torch.tensor(self.rr_label[index], dtype=torch.float32)

        inputs = torch.stack([appearance_data,motion_data],dim=0)
        targets = torch.stack([hr_target,rr_target],dim=0)

        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
            targets = targets.to('cuda')

        return inputs, targets

    def __len__(self):
        return len(self.hr_label)
