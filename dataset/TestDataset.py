import numpy as np
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import Dataset


class TestDataset(Dataset):
    def __init__(self, video_data, keypoint_data, label_data, target_length):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.video_data = video_data
        self.keypoint_data = keypoint_data
        self.label_data = label_data
        self.data_length =  len(video_data) // target_length
        self.target_length = target_length

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        video_data = self.video_data[index*self.target_length : (index+1) * self.target_length]
        label_data = self.label_data[index*self.target_length : (index+1) * self.target_length]

        keypoint_data = self.keypoint_data[index*self.target_length : (index+1) * self.target_length].astype(np.int32)
        keypoint_data = np.mean(keypoint_data,axis=0,dtype=np.int32)
        x_p = keypoint_data[::2]
        y_p = keypoint_data[1::2]

        forehead_data = video_data[:,y_p[0]:y_p[1],x_p[0]:x_p[1]]
        lcheek_data = video_data[:, y_p[2]:y_p[3],x_p[2]:x_p[3]]
        rcheek_data = video_data[:, y_p[4]:y_p[5],x_p[4]:x_p[5]]

        forehead_data = np.resize(forehead_data,(self.target_length,30,30,3))
        lcheek_data = np.resize(lcheek_data,(self.target_length,30,30,3))
        rcheek_data = np.resize(rcheek_data,(self.target_length,30,30,3))

        forehead_data = torch.tensor(np.transpose(forehead_data, (3, 0, 1, 2)), dtype=torch.float32)
        lcheek_data = torch.tensor(np.transpose(lcheek_data, (3, 0, 1, 2)), dtype=torch.float32)
        rcheek_data = torch.tensor(np.transpose(rcheek_data, (3, 0, 1, 2)), dtype=torch.float32)
        label_data = torch.tensor(label_data, dtype=torch.float32)

        if torch.cuda.is_available():
            forehead_data = forehead_data.to('cuda')
            lcheek_data = lcheek_data.to('cuda')
            rcheek_data = rcheek_data.to('cuda')
            label_data = label_data.to('cuda')

        return (forehead_data,lcheek_data,rcheek_data), label_data

    def __len__(self):
        return self.data_length
