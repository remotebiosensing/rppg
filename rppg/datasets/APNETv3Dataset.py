import numpy as np
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from skimage.transform import resize

def load_video_frames(frames):
    # Load the video frames and convert to RGB pixel values
    frames = [transforms.ToTensor()(frame) for frame in frames]
    frames = torch.stack(frames)

    return frames


class APNETv3Dataset(Dataset):
    def __init__(self, video_data, label_data, target_length):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.video_data = np.reshape(video_data, (-1, target_length, 64, 64, 3))
        self.label_data = np.reshape(label_data, (-1, target_length))

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        video_data = torch.tensor(np.transpose(self.video_data[index], (3, 0, 1, 2)), dtype=torch.float32)
        label_data = torch.tensor(self.label_data[index], dtype=torch.float32)

        if torch.cuda.is_available():
            video_data = video_data.to('cuda')
            label_data = label_data.to('cuda')

        return video_data, label_data

    def __len__(self):
        return len(self.label_data)

