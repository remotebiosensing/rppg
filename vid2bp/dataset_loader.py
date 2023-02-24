import os
import h5py
import numpy as np
from torch.utils.data import DataLoader
from vid2bp.BPNetDataset import BPNetDataset

import torch.nn as nn
import torch
from scipy import signal
import matplotlib.pyplot as plt


def dataset_loader(dataset_name, in_channel, batch_size, device, gender, normalized):
    train_shuffle: bool = True
    test_shuffle: bool = True

    if normalized:
        dataset_root_path: str = '/home/paperc/PycharmProjects/dataset/BPNet_' + dataset_name + '/additional2_202321_normalized/'
    else:
        dataset_root_path: str = '/home/paperc/PycharmProjects/dataset/BPNet_' + dataset_name + '/additional2_202321/'
    train_file_path = dataset_root_path + "train_" + gender + "_09.hdf5"
    valid_file_path = dataset_root_path + "val_" + gender + "_09.hdf5"
    test_file_path = dataset_root_path + "test_" + gender + "_09.hdf5"
    if os.path.isfile(train_file_path) and os.path.isfile(valid_file_path) and os.path.isfile(test_file_path):
        print("datasets exist")
    else:
        print("preprocessing needed... run mimic3temp.py")
    with h5py.File(train_file_path, 'r') as train_data:
        train_ple, train_abp, train_size, train_info, train_ohe = np.array(train_data['ple'][:, in_channel, :]), \
            np.array(train_data['abp']), np.array(train_data['size']), np.array(train_data['info']), np.array(
            train_data['ohe'])
    with h5py.File(valid_file_path, 'r') as valid_data:
        valid_ple, valid_abp, valid_size, valid_info, valid_ohe = np.array(valid_data['ple'][:, in_channel, :]), \
            np.array(valid_data['abp']), np.array(valid_data['size']), np.array(valid_data['info']), np.array(
            valid_data['ohe'])
    with h5py.File(test_file_path, 'r') as test_data:
        test_ple, test_abp, test_size, test_info, test_ohe = np.array(test_data['ple'][:, in_channel, :]), \
            np.array(test_data['abp']), np.array(test_data['size']), np.array(test_data['info']), np.array(
            test_data['ohe'])

    train_dataset = BPNetDataset(train_ple, train_abp, train_size, train_info, train_ohe, device)
    valid_dataset = BPNetDataset(valid_ple, valid_abp, valid_size, valid_info, valid_ohe, device)
    test_dataset = BPNetDataset(test_ple, test_abp, test_size, test_info, test_ohe, device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=test_shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=test_shuffle)

    return [train_loader, valid_loader, test_loader]

# dataset_loader(dataset_name='mimiciii', channel=3)
