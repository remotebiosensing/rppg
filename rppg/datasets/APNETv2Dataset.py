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


class APNETv2Dataset(Dataset):
    def __init__(self, video_data, keypoint_data, label_data, target_length, img_size):
        self.transform = transforms.Compose([
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        self.label_data = np.reshape(label_data,(-1,target_length))
        keypoint_data = np.clip(keypoint_data,0,img_size)
        keypoint_data = np.reshape(keypoint_data,(-1,target_length,12))
        self.target_length = target_length

        '''
        * remove flipped video 
        '''
        idx =[]
        for i in range(len(keypoint_data)):
            k = keypoint_data[i].astype(np.int32)
            k = np.mean(k, axis=0, dtype=np.int32)
            x_p = k[::2]
            y_p = k[1::2]
            if y_p[0] >= y_p[1] or y_p[2] >= y_p[3] or y_p[4] >= y_p[5] or x_p[0] >=x_p[1] or x_p[2] >=x_p[3] or x_p[4] >= x_p[5]:
                idx.append(i)

        video_data = np.delete(video_data,idx, axis = 0)
        self.label_data = np.delete(self.label_data, idx, axis = 0)
        keypoint_data = np.delete(keypoint_data, idx, axis = 0)

        # up sampling label data

        '''
        * clip roi region
        '''
        self.forehead_data = np.empty((len(video_data),self.target_length,30,30,3))
        self.lcheek_data = np.empty((len(video_data),self.target_length,30,30,3))
        self.rcheek_data = np.empty((len(video_data),self.target_length,30,30,3))


        for i in range(len(video_data)):
            # v = video_data[i]
            k = keypoint_data[i].astype(np.int32)
            k = np.mean(k, axis=0, dtype=np.int32)
            x_p = k[::2]
            y_p = k[1::2]

            f = video_data[i,:, y_p[0]:y_p[1], x_p[0]:x_p[1]]
            l = video_data[i,:, y_p[2]:y_p[3], x_p[2]:x_p[3]]
            r = video_data[i,:, y_p[4]:y_p[5], x_p[4]:x_p[5]]

            self.forehead_data[i] = resize(f[:], (self.target_length, 30, 30))
            self.lcheek_data[i] = resize(l[:], (self.target_length, 30, 30))
            self.rcheek_data[i] = resize(r[:], (self.target_length, 30, 30))


    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        forehead_data = load_video_frames(self.forehead_data[index])
        lcheek_data = load_video_frames(self.lcheek_data[index])
        rcheek_data = load_video_frames(self.rcheek_data[index])

        # if self.transform:
        #     forehead_data = self.transform(forehead_data)
        #     lcheek_data = self.transform(lcheek_data)
        #     rcheek_data = self.transform(rcheek_data)

        label_data = self.label_data[index]

        label_data = torch.tensor(label_data, dtype=torch.float32)

        if torch.cuda.is_available():
            forehead_data = forehead_data.to('cuda',dtype=torch.float32)
            lcheek_data = lcheek_data.to('cuda',dtype=torch.float32)
            rcheek_data = rcheek_data.to('cuda',dtype=torch.float32)
            label_data = label_data.to('cuda',dtype=torch.float32)

        return (forehead_data, lcheek_data, rcheek_data), label_data

    def __len__(self):
        return len(self.forehead_data)

