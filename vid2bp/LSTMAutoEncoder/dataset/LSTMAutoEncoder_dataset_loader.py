import os
import h5py
import numpy as np
from torch.utils.data import DataLoader

from vid2bp.LSTMAutoEncoder.dataset.LSTMAutoEncoderDataset import LSTMAutoEncoderDataset


def dataset_loader(batch_size: int = 8, label: str = 'ple',
                   dataset_root_path: str = '/home/najy/PycharmProjects/vid2bp_datasets/raw/'):
    train_shuffle: bool = True
    test_shuffle: bool = False

    train_file_path = dataset_root_path + "train.hdf5"
    valid_file_path = dataset_root_path + "val.hdf5"
    test_file_path = dataset_root_path + "test.hdf5"
    if os.path.isfile(train_file_path) and os.path.isfile(valid_file_path) and os.path.isfile(test_file_path):
        print("dataset exist")
    else:
        print("preprocessing needed")

    train_ple_std = 0
    train_abp_std = 0
    with h5py.File(train_file_path, 'r') as train_data:
        train_ple = np.array(train_data['ple'])
        for x in train_ple[:, 0]:
            train_ple_std += np.std(x)
        train_ple_std /= len(train_ple)
        train_ple_mean = np.mean(train_ple)
        train_ple[:, 0] = (train_ple[:, 0] - train_ple_mean) / train_ple_std

        train_abp = np.array(train_data['abp'])
        for x in train_abp:
            train_abp_std += np.std(x)
        train_abp_std /= len(train_abp)
        train_abp_mean = np.mean(train_abp)
        train_abp = (train_abp - train_abp_mean) / train_abp_std

    valid_ple_std = 0
    valid_abp_std = 0
    with h5py.File(valid_file_path, 'r') as valid_data:
        valid_ple = np.array(valid_data['ple'])
        for x in valid_ple[:, 0]:
            valid_ple_std += np.std(x)
        valid_ple_std /= len(valid_ple)
        valid_ple_mean = np.mean(valid_ple)
        valid_ple[:, 0] = (valid_ple[:, 0] - valid_ple_mean) / valid_ple_std
        valid_abp = np.array(valid_data['abp'])
        for x in valid_abp:
            valid_abp_std += np.std(x)
        valid_abp_std /= len(valid_abp)
        valid_abp_mean = np.mean(valid_abp)
        valid_abp = (valid_abp - valid_abp_mean) / valid_abp_std

    test_ple_std = 0
    test_abp_std = 0
    with h5py.File(test_file_path, 'r') as test_data:
        test_ple = np.array(test_data['ple'])
        for x in test_ple[:, 0]:
            test_ple_std += np.std(x)
        test_ple_std /= len(test_ple)
        test_ple_mean = np.mean(test_ple)
        test_ple[:, 0] = (test_ple[:, 0] - test_ple_mean) / test_ple_std

        test_abp = np.array(test_data['abp'])
        for x in test_abp:
            test_abp_std += np.std(x)
        test_abp_std /= len(test_abp)
        test_abp_mean = np.mean(test_abp)
        test_abp = (test_abp - test_abp_mean) / test_abp_std

    if label == 'ple':
        train_dataset = LSTMAutoEncoderDataset(train_ple, train_ple[:, 0], train_ple_mean, train_ple_std)
        valid_dataset = LSTMAutoEncoderDataset(valid_ple, valid_ple[:, 0], valid_ple_mean, valid_ple_std)
        test_dataset = LSTMAutoEncoderDataset(test_ple, test_ple[:, 0], test_ple_mean, test_ple_std)
    elif label == 'abp':
        train_dataset = LSTMAutoEncoderDataset(train_ple, train_abp, train_abp_mean, train_abp_std)
        valid_dataset = LSTMAutoEncoderDataset(valid_ple, valid_abp, valid_abp_mean, valid_abp_std)
        test_dataset = LSTMAutoEncoderDataset(test_ple, test_abp, test_abp_mean, test_abp_std)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=test_shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=test_shuffle)

    return [train_loader, valid_loader, test_loader], [train_ple_std, train_ple_mean]
