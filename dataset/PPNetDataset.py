import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class PPNetDataset(Dataset):
    def __init__(self, ppg, sbp, dbp, hr):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.ppg = ppg
        self.sbp = sbp
        self.dbp = dbp
        self.hr = hr

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()


        ppg = torch.tensor(self.ppg[index], dtype=torch.float32)
        sbp = torch.tensor(self.sbp[index], dtype=torch.float32)
        dbp = torch.tensor(self.dbp[index], dtype=torch.float32)
        hr = torch.tensor(self.hr[index], dtype=torch.float32)

        inputs = ppg.reshape(1,-1)
        targets = torch.stack([sbp,dbp,hr],dim=0)

        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
            targets = targets.to('cuda')

        return inputs, targets

    def __len__(self):
        return len(self.ppg)
