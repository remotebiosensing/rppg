import os
import h5py
import numpy as np
from torch.utils.data import DataLoader
from vid2bp.BPNetDataset import BPNetDataset

import torch.nn as nn
import torch
from scipy import signal
import matplotlib.pyplot as plt

def dataset_loader(dataset_name: str = 'mimiciii', channel: int = 1, batch_size: int = 512):
    train_shuffle: bool = True
    test_shuffle: bool = True

    upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)


    # dataset_root_path: str = '/home/paperc/PycharmProjects/dataset/BPNet_' + dataset_name + '/0112/'
    # train_file_path = dataset_root_path + "train.hdf5"
    # valid_file_path = dataset_root_path + "val.hdf5"
    # test_file_path = dataset_root_path + "test.hdf5"
    # if os.path.isfile(train_file_path) and os.path.isfile(valid_file_path) and os.path.isfile(test_file_path):
    #     print("datasets exist")
    # else:
    #     print("preprocessing needed")
    # with h5py.File(train_file_path, 'r') as train_data:
    #     train_ple, train_abp, train_size = np.array(train_data['ple'][:, :channel, :]), \
    #         np.array(train_data['abp']), np.array(train_data['size'])
    # with h5py.File(valid_file_path, 'r') as valid_data:
    #     valid_ple, valid_abp, valid_size = np.array(valid_data['ple'][:, :channel, :]), \
    #         np.array(valid_data['abp']), np.array(valid_data['size'])
    # with h5py.File(test_file_path, 'r') as test_data:
    #     test_ple, test_abp, test_size = np.array(test_data['ple'][:, :channel, :]), \
    #         np.array(test_data['abp']), np.array(test_data['size'])

    # bpnet 1.0
    dataset_root_path: str = '/home/paperc/PycharmProjects/dataset/BPNet_uci/case(P+V+A)_750_train(cv1).hdf5'
    with h5py.File(dataset_root_path, 'r') as dataset:
        train_ple, train_abp, train_size = np.array(dataset['train']['ple']['0'][:, :channel, :]), \
            np.array(dataset['train']['abp']['0']), np.array(dataset['train']['size']['0'])
        valid_ple, valid_abp, valid_size = np.array(dataset['validation']['ple']['0'][:, :channel, :]), \
            np.array(dataset['validation']['abp']['0']), np.array(dataset['validation']['size']['0'])
        test_ple, test_abp, test_size = np.array(dataset['validation']['ple']['0'][:, :channel, :]), \
            np.array(dataset['validation']['abp']['0']), np.array(dataset['validation']['size']['0'])

    train_dataset = BPNetDataset(train_ple[:171008], train_abp[:171008], train_size[:171008])
    # train_dataset = BPNetDataset(train_ple, train_abp, train_size)
    valid_dataset = BPNetDataset(valid_ple[:41984], valid_abp[:41984], valid_size[:41984])
    # valid_dataset = BPNetDataset(valid_ple, valid_abp, valid_size)
    test_dataset = BPNetDataset(test_ple[:41984], test_abp[:41984], test_size[:41984])
    # test_dataset = BPNetDataset(test_ple, test_abp, test_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle)
    # valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=test_shuffle)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=test_shuffle)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=test_shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=test_shuffle)

    return [train_loader, valid_loader, test_loader]

# dataset_loader(dataset_name='mimiciii', channel=3)
