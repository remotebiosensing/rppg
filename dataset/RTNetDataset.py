import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class RTNetDataset(Dataset):
    def __init__(self, face_data, mask_data, target):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.face = face_data
        self.mask = mask_data
        self.label = target.reshape(-1, 1)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        appearance_data = torch.tensor(np.transpose(self.face[index], (2, 0, 1)), dtype=torch.float32)
        motion_data = torch.tensor(np.transpose(self.mask[index], (2, 0, 1)), dtype=torch.float32)
        target = torch.tensor(self.label[index], dtype=torch.float32)

        inputs = torch.stack([appearance_data,motion_data],dim=0)

        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
            target = target.to('cuda')

        return inputs, target

    def __len__(self):
        return len(self.label)
