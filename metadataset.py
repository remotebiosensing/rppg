import random

import h5py
import numpy as np
import scipy.io
from scipy.signal import butter
from torchmeta.utils.data import Task, MetaDataset

from spliiter import ClassSplitter

np.random.seed(100)

class RPPG_DATASET(MetaDataset):
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
    def __init__(self, dataset, num_shots_tr, num_shots_ts, person_data_path, num_tasks=1000000, state='train',
                 transform=None, target_transform=None, sample_type='person', random_seed=10,
                 frame_depth=10, fs=30, signal='pulse', unsupervised=0):
        super(RPPG_DATASET, self).__init__(meta_split='train', target_transform=target_transform)
        self.num_samples_per_task = num_shots_tr + num_shots_ts
        self.num_tasks = num_tasks
        self.person_data_path = person_data_path
        self.target_transform = target_transform
        self.transform = transform
        self.sample_type = sample_type
        self.dataset = dataset
        self.frame_depth = frame_depth
        self.fs = fs
        self.state = state
        self.num_shots_tr = num_shots_tr
        self.signal = signal
        self.unsupervised = unsupervised
        np.random.seed(random_seed)
        if self.state == 'train':
            self.dataset_transform = ClassSplitter(shuffle=False, num_train_per_class=num_shots_tr,
                                                   num_test_per_class=num_shots_ts)

    def __len__(self):
        return self.num_tasks

    def _diverse_sampling(self, sample_paths, num_shots, sample_type='task'):
        sample_id = []
        final_paths = []
        cnt = 1
        while len(final_paths) < num_shots:
            # reset every 12 files added
            if len(sample_id) == 12:
                sample_id = []
                cnt = 1
            temp_path = random.sample(sample_paths, 1)[0]
            file_name = temp_path.split('/')[-1].split('.')[0]
            if sample_type == 'task':
                first_index = file_name.find('T')
                last_index = file_name.find('V')
            elif sample_type == 'person':
                first_index = file_name.find('P')
                last_index = file_name.find('T')
            else:
                raise ValueError('The sample type is not supported!')

            id_temp = file_name[first_index + 1:last_index]
            if (id_temp not in sample_id) and (int(id_temp) == cnt) and (temp_path not in final_paths):
                sample_id.append(id_temp)
                final_paths.append(temp_path)
                cnt += 1
        return final_paths

    def _hard_sampling(self, sample_paths, num_shots, sample_type='task'):
        final_paths = []
        if num_shots == 20:
            while len(final_paths) < num_shots:
                # reset every 12 files added
                temp_path = random.sample(sample_paths, 1)[0]
                file_name = temp_path.split('/')[-1].split('.')[0]
                if sample_type == 'task':
                    first_index = file_name.find('T')
                    last_index = file_name.find('V')
                elif sample_type == 'person':
                    first_index = file_name.find('P')
                    last_index = file_name.find('T')
                else:
                    raise ValueError('The sample type is not supported!')

                id_temp = file_name[first_index + 1:last_index]
                if int(id_temp) in [6, 12] and (temp_path not in final_paths):
                    final_paths.append(temp_path)
        else:
            while len(final_paths) < num_shots:
                # reset every 12 files added
                temp_path = random.sample(sample_paths, 1)[0]
                file_name = temp_path.split('/')[-1].split('.')[0]
                if sample_type == 'task':
                    first_index = file_name.find('T')
                    last_index = file_name.find('V')
                elif sample_type == 'person':
                    first_index = file_name.find('P')
                    last_index = file_name.find('T')
                else:
                    raise ValueError('The sample type is not supported!')

                id_temp = file_name[first_index + 1:last_index]
                if (int(id_temp) in [6, 12] and (temp_path not in final_paths)) or \
                        (len(final_paths) >= 10 and (temp_path not in final_paths)):
                    final_paths.append(temp_path)
        return final_paths

    def __getitem__(self, index):
        per_task_data = self.person_data_path[index]
        # task_path = self._diverse_sampling(per_task_data, self.num_samples_per_task, self.sample_type)
        # task_path = self._hard_sampling(per_task_data, self.num_samples_per_task, self.sample_type)
        # if self.num_samples_per_task == 20:
        #    print('task_path: ', task_path)
        #    print('=============================================')

        if self.state == 'test':
            self.num_shots_ts = len(per_task_data) - self.num_shots_tr
            self.dataset_transform = ClassSplitter(shuffle=False, num_train_per_class=self.num_shots_tr,
                                                   num_test_per_class=self.num_shots_ts)
            self.num_samples_per_task = self.num_shots_tr + self.num_shots_ts

        if self.state == 'train':
            random.shuffle(per_task_data)
            if 'AFRL' in per_task_data[0]:
                self.num_shots_ts = 8
                self.dataset_transform = ClassSplitter(shuffle=False, num_train_per_class=self.num_shots_tr,
                                                                           num_test_per_class=self.num_shots_ts)
                self.num_samples_per_task = self.num_shots_tr + self.num_shots_ts
            else:
                self.num_shots_ts = len(per_task_data) - self.num_shots_tr
                if self.num_shots_ts > 8:
                    self.num_shots_ts = 8
                self.dataset_transform = ClassSplitter(shuffle=False, num_train_per_class=self.num_shots_tr,
                                                                           num_test_per_class=self.num_shots_ts)
                self.num_samples_per_task = self.num_shots_tr + self.num_shots_ts

        task_path = per_task_data[:self.num_samples_per_task]
        task = PersonTask(self.num_samples_per_task, task_path, self.num_shots_tr, self.dataset, self.transform,
                          self.target_transform, frame_depth=self.frame_depth, fs=self.fs, state=self.state,
                          signal=self.signal, unsupervised=self.unsupervised)

        if self.dataset_transform is not None:
            task = self.dataset_transform(task)
        return task



class PersonTask(Task):
    def __init__(self, num_samples, task_data_path, num_shots_tr, dataset, transform=None, target_transform=None, frame_depth=10,
                 fs=30, state='train', signal='pulse', unsupervised=0):
        super(PersonTask, self).__init__(None, None) # Regression task
        self.num_shots_tr = num_shots_tr
        self.num_samples = num_samples
        self.transform = transform
        self.target_transform = target_transform
        self.task_data_path = task_data_path
        self.dataset = dataset
        self.frame_depth = frame_depth
        self.fs = fs
        self.state = state
        self.signal = signal
        self.unsupervised = unsupervised

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        temp_path = self.task_data_path[index]
        if self.dataset == 'MAHNOB-HCI':
            f1 = scipy.io.loadmat(temp_path)
        else:
            f1 = h5py.File(temp_path, 'r')
        output = np.transpose(np.array(f1["dXsub"]), [3, 0, 2, 1])
        if self.unsupervised == 1 and self.state == 'train':
            label = np.array(f1["dssub"])
        elif self.unsupervised == 1 and self.state == 'test' and index < self.num_shots_tr:
            label = np.array(f1["dssub"])
        else:
            label = np.array(f1["dysub"])

        if 'AFRL' in temp_path:
            self.fs = 30
            if self.signal == 'pulse':
                [b, a] = butter(1, [0.75 / self.fs * 2, 2.5 / self.fs * 2], btype='bandpass')
            else:
                # Resp filter
                label = np.array(f1["drsub"])
                [b, a] = butter(1, [0.08 / self.fs * 2, 0.5 / self.fs * 2], btype='bandpass')
            label = scipy.signal.filtfilt(b, a, np.squeeze(label))
            label = np.expand_dims(label, axis=1)
        elif 'MMSE' in temp_path:
            self.fs = 25
            [b, a] = butter(1, [0.75 / self.fs * 2, 2.5 / self.fs * 2], btype='bandpass')
            label = scipy.signal.filtfilt(b, a, np.squeeze(label))
            label = np.expand_dims(label, axis=1)
        else:
            self.fs = 30
            [b, a] = butter(1, [0.75 / self.fs * 2, 2.5 / self.fs * 2], btype='bandpass')
            label = scipy.signal.filtfilt(b, a, np.squeeze(label))
            label = np.expand_dims(label, axis=1)
        # Average the frame
        motion_data = output[:, :3, :, :]
        apperance_data = output[:, 3:, :, :]
        apperance_data = np.reshape(apperance_data, (int(180/self.frame_depth), self.frame_depth, 3, 36, 36))
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