import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class LSTMAutoEncoderDataset(Dataset):
    def __init__(self, ppg, abp, mean, std):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.ppg = ppg
        self.abp = abp
        self.mean = mean
        self.std = std

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        ppg = torch.tensor(self.ppg[index], dtype=torch.float32).transpose(1, 0)
        abp = torch.tensor(self.abp[index], dtype=torch.float32)

        if torch.cuda.is_available():
            ppg = ppg.to('cuda')
            abp = abp.to('cuda')

        return ppg, abp

    def __len__(self):
        return len(self.ppg)
