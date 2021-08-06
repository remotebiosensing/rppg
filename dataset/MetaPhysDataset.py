import random

import h5py
import numpy as np
import scipy.io
from scipy.signal import butter
from torchmeta.utils.data import Task, MetaDataset
import torchvision.transforms as transforms

from utils.Meta_class_splitters import ClassSplitter
from utils.funcs import ToTensor1D

np.random.seed(100)

class MetaPhysDataset(MetaDataset):
    """
    Simple regression task, based on sinusoids, as introduced in [1].

    Parameters
    ----------
    num_samples_per_task : int
        Number of examples per task.

    num_tasks : int (default: 1,000,000)
        Overall number of tasks to sample.

    transform : callable, optional
        A function/transform that takes a numpy array of size (1,) and returns a
        transformed version of the input.

    target_transform : callable, optional
        A function/transform that takes a numpy array of size (1,) and returns a
        transformed version of the target.

    dataset_transform : callable, optional
        A function/transform that takes a dataset (ie. a task), and returns a
        transformed version of it. E.g. `torchmeta.transforms.ClassSplitter()`.
    """
    def __init__(self, num_shots_tr, num_shots_ts, person_data_path, state='train',transform=None,
                 target_transform= None, random_seed=10, frame_depth=10, fs=30, unsupervised=0):
        super(MetaPhysDataset, self).__init__(meta_split='train', target_transform=ToTensor1D())
        self.transform = ToTensor1D()
        self.num_samples_per_task = num_shots_tr + num_shots_ts
        self.person_data_path = person_data_path
        self.frame_depth = frame_depth
        self.fs = fs
        self.state = state
        self.num_shots_tr = num_shots_tr
        self.unsupervised = unsupervised

        self.target_transform = target_transform
        self.transform = transform

        np.random.seed(random_seed)
        if self.state == 'train':
            self.dataset_transform = ClassSplitter(shuffle=False, num_train_per_class=num_shots_tr,
                                                   num_test_per_class=num_shots_ts)

    def __len__(self):
        return len(self.person_data_path)


    def __getitem__(self, index):
        per_task_data = self.person_data_path[index][:-1]

        if self.state == 'test':
            self.num_shots_ts = len(per_task_data) - self.num_shots_tr
            self.dataset_transform = ClassSplitter(shuffle=False, num_train_per_class=self.num_shots_tr,
                                                   num_test_per_class=self.num_shots_ts)
            self.num_samples_per_task = self.num_shots_tr + self.num_shots_ts

        if self.state == 'train':
            random.shuffle(per_task_data)

            self.num_shots_ts = len(per_task_data) - self.num_shots_tr
            if self.num_shots_ts > 8:
                self.num_shots_ts = 8
            self.dataset_transform = ClassSplitter(shuffle=False, num_train_per_class=self.num_shots_tr,
                                                                       num_test_per_class=self.num_shots_ts)
            self.num_samples_per_task = self.num_shots_tr + self.num_shots_ts

        task_path = per_task_data[:self.num_samples_per_task]
        task = PersonTask(self.num_samples_per_task, task_path, self.num_shots_tr, frame_depth=self.frame_depth, fs=self.fs, state=self.state, unsupervised=self.unsupervised)

        if self.dataset_transform is not None:
            task = self.dataset_transform(task)
        return task



class PersonTask(Task):
    def __init__(self, num_samples, task_data_path, num_shots_tr, frame_depth=10,
                 fs=30, state='train', unsupervised=0):
        super(PersonTask, self).__init__(None, None) # Regression task
        self.num_shots_tr = num_shots_tr
        self.num_samples = num_samples
        self.transform = ToTensor1D()
        self.target_transform = ToTensor1D()
        self.task_data_path = task_data_path
        self.frame_depth = frame_depth
        self.fs = fs
        self.state = state
        self.unsupervised = unsupervised
        self.len_data = 0

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        temp_path = self.task_data_path[index]
        f1 = h5py.File(temp_path, 'r')

        self.len_data = len(f1["preprocessed_video"]) // 10
        output = np.transpose(np.array(f1["preprocessed_video"]), [0, 3, 2, 1])[:self.len_data*10]
        label = np.array(f1["preprocessed_label"])[: self.len_data * 10]

        [b, a] = butter(1, [0.75 / self.fs * 2, 2.5 / self.fs * 2], btype='bandpass')
        label = scipy.signal.filtfilt(b, a, np.squeeze(label))
        label = np.expand_dims(label, axis=1)

        # Average the frame
        motion_data = output[:, :3, :, :]
        apperance_data = output[:, 3:, :, :]
        apperance_data = np.reshape(apperance_data, (self.len_data, self.frame_depth, 3, 36, 36))
        apperance_data = np.average(apperance_data, axis=1)
        apperance_data = np.repeat(apperance_data[:, np.newaxis, :, :, :], self.frame_depth, axis=1)
        apperance_data = np.reshape(apperance_data, (apperance_data.shape[0] * apperance_data.shape[1],
                                                     apperance_data.shape[2], apperance_data.shape[3],
                                                     apperance_data.shape[4]))
        output = np.concatenate((motion_data, apperance_data), axis=1)

        if self.transform is not None:
            output = self.transform(output)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return output, label

