import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class RhythmNetDataset(Dataset):
    """
        Dataset class for RhythmNet
    """
    # The data is now the SpatioTemporal Maps instead of videos

    def __init__(self, st_map_data, target_data):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.st_map_data = st_map_data
        self.target_data = target_data

    def __len__(self):
        return len(self.st_map_data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        maps = torch.tensor(np.transpose(self.st_map_data[index], [2, 0, 1]), dtype=torch.float32)
        # maps = torch.tensor(self.st_map_data[index], dtype=torch.float32)
        target_data = torch.tensor(self.target_data[index], dtype=torch.float32)

        if torch.cuda.is_available():
            st_map_data = maps.to('cuda')
            target_data = target_data.to('cuda')

        return st_map_data, target_data
