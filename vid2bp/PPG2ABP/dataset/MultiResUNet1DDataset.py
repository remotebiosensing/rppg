import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class MultiResUNet1DDataset(Dataset):
    def __init__(self, ppg, abp_out, abp_level1, abp_level2, abp_level3, abp_level4, max_abp, min_abp, max_ppg,
                 min_ppg):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.ppg = ppg
        self.abp_out = abp_out
        self.length = len(ppg[-1])
        self.max_abp = max_abp
        self.min_abp = min_abp
        self.max_ppg = max_ppg
        self.min_ppg = min_ppg

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        ppg = torch.unsqueeze(torch.tensor(self.ppg[index], dtype=torch.float32), 0)
        abp_out = torch.tensor(self.abp_out[index], dtype=torch.float32).transpose(1, 0)

        if torch.cuda.is_available():
            ppg = ppg.to('cuda')
            abp_out = abp_out.to('cuda')

        return ppg, abp_out

    def __len__(self):
        return len(self.ppg)

    def min_max_data(self):
        return self.max_abp, self.min_abp, self.max_ppg, self.min_ppg
