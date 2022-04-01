import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class AxisNetDataset(Dataset):
    def __init__(self, video_data, ptt_data, label_data):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.video_data = video_data
        self.ptt_data = ptt_data
        for idx,data in enumerate(label_data):
            label_data[idx] -= np.mean(label_data[idx])
            label_data[idx] /= np.std(label_data[idx])
        self.label = label_data

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()


        video_data = torch.tensor(np.transpose(self.video_data[index], (2, 0, 1)), dtype=torch.float32)
        # video_data = video_data + torch.randn(video_data.shape)*0.2
        ptt_data = torch.tensor(np.transpose(self.ptt_data[index], (2, 0, 1)), dtype=torch.float32)
        label_data = torch.tensor(self.label[index], dtype=torch.float32)

        if torch.cuda.is_available():
            video_data = video_data.to('cuda')
            ptt_data = ptt_data.to('cuda')
            label_data = label_data.to('cuda')

        return (video_data, ptt_data), label_data

    def __len__(self):
        return len(self.label)

