import random

import torch
import numpy as np
from torchmeta.utils.data import Task, MetaDataset
import torchvision.transforms as transforms
from torchmeta.transforms import ClassSplitter

import scipy.io
from scipy.signal import butter

class MetaPhysDataset(MetaDataset):
    def __init__(self, num_shots_tr, num_shots_ts, video_data, label_data,
                 option='train', unsupervised=0,frame_depth=10,
                 ):

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.num_samples_per_task = num_shots_tr + num_shots_ts
        self.frame_depth = frame_depth
        self.option = option
        self.num_shots_tr = num_shots_tr
        self.num_shots_ts = num_shots_ts
        self.unsupervised = unsupervised
        self.dataset_transform = ClassSplitter(shuffle=False, num_train_per_class=num_shots_tr,
                                                   num_test_per_class=num_shots_ts)

        self.video_data = video_data
        self.label = label_data


    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        vi = []
        la = []

        data_len = len(self.label[index]) // 2  # support set 1 + query set 1
        for i in range(2):
            vi.append(self.video_data[index][data_len * i:data_len * (i + 1)])
            la.append(self.label[index][data_len * i:data_len * (i + 1)])


        self.dataset_transform = ClassSplitter(shuffle=False, num_train_per_class=self.num_shots_tr,
                                               num_test_per_class=self.num_shots_ts)

        task = PersonTask(vi, la, len(vi))
        if self.dataset_transform is not None:
            task = self.dataset_transform(task)

        return task

    def __len__(self):
        return len(self.label)


class PersonTask(Task):
    def __init__(self, video ,label, num_samples):
        super(PersonTask, self).__init__(None, None) # Regression task
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.video = video
        self.label = label
        self.num_samples = num_samples

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):

        '''
        if index<self.num_samples:
            video_data = np.concatenate((self.video[index], self.video[index + 1]), axis=0)
            label_data = np.concatenate((self.label[index], self.label[index + 1]), axis=0)

            video_data = torch.tensor(np.transpose(video_data, (0, 4, 1, 2, 3)), dtype=torch.float32)
            label_data = torch.tensor(label_data, dtype=torch.float32)

            if torch.cuda.is_available():
                video_data = video_data.to('cuda:9')
                label_data = label_data.to('cuda:9')

            return video_data, label_data


        '''
        video_data = torch.tensor(np.transpose(self.video[index], (0, 4, 1, 2, 3)), dtype=torch.float32)
        #video_data = torch.tensor(np.transpose(self.video[index], (3, 0, 1, 2)), dtype=torch.float32)
        label_data = torch.tensor(self.label[index], dtype=torch.float32)

        if torch.cuda.is_available():
            video_data = video_data.to('cuda')
            label_data = label_data.to('cuda')

        return video_data, label_data

