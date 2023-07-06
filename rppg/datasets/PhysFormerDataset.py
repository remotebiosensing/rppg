import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset



class PhysFormerDataset(Dataset):
    def __init__(self, video_data, label_data, average_hr, target_length):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.video_data = np.reshape(video_data, (-1, target_length, video_data.shape[2], video_data.shape[3], 3))*2-1
        average_hr = [x if x > 40. else 40. for x in average_hr]
        average_hr = [x if x < 180. else 180. for x in average_hr]
        average_hr = [(x - 40.) for x in average_hr]
        self.average_hr = average_hr
        self.label_data = np.reshape(label_data, (-1, target_length))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # zscore_video_data = (self.video_data[idx] - np.mean(self.video_data[idx])) / np.std(self.video_data[idx])
        # zerotoone_video_data = (self.video_data[idx] - np.min(self.video_data[idx])) /\
        #                        (np.max(self.video_data[idx]) - np.min(self.video_data[idx]))
        # video 크기 확인@@@@
        # video_data = torch.tensor(np.transpose(zerotoone_video_data, (3, 0, 1, 2)), dtype=torch.float32)
        video_data = torch.tensor(np.transpose(self.video_data[idx], (3, 0, 1, 2)), dtype=torch.float32)
        label_data = torch.tensor(self.label_data[idx], dtype=torch.float32)
        average_hr = torch.tensor(self.average_hr[idx], dtype=torch.float32)

        if torch.cuda.is_available():
            video_data = video_data.to('cuda')
            label_data = label_data.to('cuda')
            average_hr = average_hr.to('cuda')

        return video_data, label_data, average_hr

    def __len__(self):
        return len(self.label_data)
