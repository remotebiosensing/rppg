import random

import torch
import numpy as np
from torchmeta.utils.data import Task, MetaDataset
import torchvision.transforms as transforms
from torchmeta.transforms import ClassSplitter

class MetaPhysDataset(MetaDataset):
    def __init__(self, num_shots_tr, num_shots_ts, option='train',
                 unsupervised=0,frame_depth=10,
                 appearance_data=None, motion_data=None, target=None):

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.num_samples_per_task = num_shots_tr + num_shots_ts
        self.frame_depth = frame_depth
        self.option = option
        self.num_shots_tr = num_shots_tr
        self.num_shots_ts = num_shots_ts
        self.unsupervised = unsupervised
        self.a = appearance_data
        self.m = motion_data
        self.label = target
        self.dataset_transform = ClassSplitter(shuffle=False, num_train_per_class=num_shots_tr,
                                                   num_test_per_class=num_shots_ts)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        ap = []
        mo = []
        la = []
        data_len = len(self.label[index]) // self.num_samples_per_task  # 1개의 데이터를 8개로
        for i in range(self.num_samples_per_task):
            ap.append(self.a[index][data_len * i:data_len * (i + 1)])
            mo.append(self.m[index][data_len * i:data_len * (i + 1)])
            la.append(self.label[index][data_len * i:data_len * (i + 1)])

        task = PersonTask(ap, mo, la)

        self.dataset_transform = ClassSplitter(shuffle=False, num_train_per_class=self.num_shots_tr,
                                               num_test_per_class=self.num_shots_ts)
        if self.dataset_transform is not None:
            task = self.dataset_transform(task)

        return task

    def __len__(self):
        return len(self.label)

class MetaPhys_task_Dataset(MetaDataset):
    def __init__(self, num_shots_tr, num_shots_ts, option='train',
                 unsupervised=0,frame_depth=10,
                 appearance_data=None, motion_data=None, target=None):

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.num_samples_per_task = num_shots_tr + num_shots_ts
        self.frame_depth = frame_depth
        self.option = option
        self.num_shots_tr = num_shots_tr
        self.num_shots_ts = num_shots_ts
        self.unsupervised = unsupervised
        self.a = appearance_data
        self.m = motion_data
        self.label = target
        self.dataset_transform = ClassSplitter(shuffle=False, num_train_per_class=num_shots_tr,
                                                   num_test_per_class=num_shots_ts)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        ap = []
        mo = []
        la = []

        for i in range(4):
            ap.append(self.a[index][i])
            mo.append(self.m[index][i])
            la.append(self.label[index][i])

        task = PersonTask(ap, mo, la)

        self.dataset_transform = ClassSplitter(shuffle=False, num_train_per_class=self.num_shots_tr,
                                               num_test_per_class=self.num_shots_ts)
        if self.dataset_transform is not None:
            task = self.dataset_transform(task)

        return task

    def __len__(self):
        return len(self.label)

class PersonTask(Task):
    def __init__(self, a, m ,label):
        super(PersonTask, self).__init__(None, None) # Regression task
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.a = a
        self.m = m
        self.label = label
        self.len_data = 0

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        self.len_data = len(self.label[index]) // 10
        appearance_data = torch.tensor(np.transpose(self.a[index], (0, 3, 2, 1)), dtype=torch.float32)[:self.len_data*10]
        motion_data = torch.tensor(np.transpose(self.m[index], (0, 3, 2, 1)), dtype=torch.float32)[:self.len_data*10]

        target = torch.tensor(self.label[index], dtype=torch.float32)[:self.len_data*10]
        input = torch.cat([appearance_data, motion_data], dim=1)

        if torch.cuda.is_available():
            input = input.to('cuda:9')
            target = target.to('cuda:9')

        return input, target