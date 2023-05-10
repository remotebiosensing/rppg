import os
import h5py
import numpy as np
from torch.utils.data import DataLoader
from cnibp.datasets.BPNetDataset import BPNetDataset


def dataset_loader(dataset_name, in_channel, batch_size, device, gender):
    train_shuffle: bool = True
    test_shuffle: bool = True

    # dataset_root_path: str = '/home/paperc/PycharmProjects/dataset/BPNet_' + dataset_name + '/additional_2023223/'
    dataset_root_path: str = '/hdd/hdd1/dataset/bpnet/preprocessed_2023418_restored/total/'
    train_file_path = dataset_root_path + "Train_" + gender + "_0.9.hdf5"
    valid_file_path = dataset_root_path + "Val_" + gender + "_0.9.hdf5"
    test_file_path = dataset_root_path + "Test_" + gender + "_0.9.hdf5"
    if os.path.isfile(train_file_path) and os.path.isfile(valid_file_path) and os.path.isfile(test_file_path):
        print('using ', dataset_root_path)
        print("datasets exist")
    else:
        print("preprocessing needed... run mimic3temp.py")
    with h5py.File(train_file_path, 'r') as train_data:
        train_ple, train_ple_cycle, train_abp, train_abp_cycle, \
        train_dbp, train_sbp, train_info = np.array(train_data['ple']), \
                                           np.array(train_data['ple_cycle']), \
                                           np.array(train_data['abp']), \
                                           np.array(train_data['abp_cycle']), \
                                           np.array(train_data['dbp']), \
                                           np.array(train_data['sbp']), \
                                           np.array(train_data['info'])
    with h5py.File(valid_file_path, 'r') as valid_data:
        valid_ple, valid_ple_cycle, valid_abp, valid_abp_cycle, \
        valid_dbp, valid_sbp, valid_info = np.array(valid_data['ple']), \
                                           np.array(valid_data['ple_cycle']), \
                                           np.array(valid_data['abp']), \
                                           np.array(valid_data['abp_cycle']), \
                                           np.array(valid_data['dbp']), \
                                           np.array(valid_data['sbp']), \
                                           np.array(valid_data['info'])
        # valid_ple, valid_abp, valid_size, valid_info, valid_ohe = np.array(valid_data['ple'][:, in_channel, :]), \
        #                                                           np.array(valid_data['abp']), np.array(
        #     valid_data['size']), np.array(valid_data['info']), np.array(
        #     valid_data['ohe'])
    with h5py.File(test_file_path, 'r') as test_data:
        test_ple, test_ple_cycle, test_abp, test_abp_cycle, \
        test_dbp, test_sbp, test_info = np.array(test_data['ple']), \
                                        np.array(test_data['ple_cycle']), \
                                        np.array(test_data['abp']), \
                                        np.array(test_data['abp_cycle']), \
                                        np.array(test_data['dbp']), \
                                        np.array(test_data['sbp']), \
                                        np.array(test_data['info'])
        # test_ple, test_abp, test_size, test_info, test_ohe = np.array(test_data['ple'][:, in_channel, :]), \
        #                                                      np.array(test_data['abp']), np.array(
        #     test_data['size']), np.array(test_data['info']), np.array(
        #     test_data['ohe'])

    train_dataset = BPNetDataset(train_ple, train_ple_cycle, train_abp, train_abp_cycle,
                                 train_dbp, train_sbp, train_info, device)
    valid_dataset = BPNetDataset(valid_ple, valid_ple_cycle, valid_abp, valid_abp_cycle,
                                 valid_dbp, valid_sbp, valid_info, device)
    test_dataset = BPNetDataset(test_ple, test_ple_cycle, test_abp, test_abp_cycle,
                                test_dbp, test_sbp, test_info, device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=test_shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=test_shuffle)

    return [train_loader, valid_loader, test_loader]

# dataset_loader(dataset_name='mimiciii', channel=3)
