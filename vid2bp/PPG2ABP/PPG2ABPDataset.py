import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class PPG2ABPDataset(Dataset):
    def __init__(self, ppg, abp_out, abp_level1, abp_level2, abp_level3, abp_level4):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.ppg = ppg
        self.abp_out = abp_out
        self.abp_level1 = abp_level1
        self.abp_level2 = abp_level2
        self.abp_level3 = abp_level3
        self.abp_level4 = abp_level4

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        ppg = torch.unsqueeze(torch.tensor(self.ppg[index], dtype=torch.float32), 0)
        abp_out = torch.tensor(self.abp_out[index], dtype=torch.float32).transpose(1, 0)
        abp_level1 = torch.tensor(self.abp_level1[index], dtype=torch.float32).transpose(1, 0)
        abp_level2 = torch.tensor(self.abp_level2[index], dtype=torch.float32).transpose(1, 0)
        abp_level3 = torch.tensor(self.abp_level3[index], dtype=torch.float32).transpose(1, 0)
        abp_level4 = torch.tensor(self.abp_level4[index], dtype=torch.float32).transpose(1, 0)
        # abp = torch.cat((abp_out, abp_level1, abp_level2, abp_level3, abp_level4)).transpose(1, 0)

        if torch.cuda.is_available():
            ppg = ppg.to('cuda')
            # abp = abp.to('cuda')
            abp_out = abp_out.to('cuda')
            abp_level1 = abp_level1.to('cuda')
            abp_level2 = abp_level2.to('cuda')
            abp_level3 = abp_level3.to('cuda')
            abp_level4 = abp_level4.to('cuda')

        return ppg, abp_out, abp_level1, abp_level2, abp_level3, abp_level4

    def __len__(self):
        return len(self.ppg)
