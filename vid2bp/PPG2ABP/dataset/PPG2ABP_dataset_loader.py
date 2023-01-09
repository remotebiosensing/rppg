import os
import h5py
import numpy as np
from torch.utils.data import DataLoader

from vid2bp.PPG2ABP.dataset.UNetDS64Dataset import PPG2ABPDataset
from vid2bp.PPG2ABP.dataset.MultiResUNet1DDataset import MultiResUNet1DDataset


def dataset_loader(dataset_name: str = 'mimiciii', channel: int = 1, batch_size: int = 8):
    train_shuffle: bool = True
    test_shuffle: bool = False

    dataset_root_path: str = '/home/najy/PycharmProjects/PPG2ABP_datasets/'
    train_file_path = dataset_root_path + "train.hdf5"
    valid_file_path = dataset_root_path + "val.hdf5"
    test_file_path = dataset_root_path + "test.hdf5"
    if os.path.isfile(train_file_path) and os.path.isfile(valid_file_path) and os.path.isfile(test_file_path):
        print("dataset exist")
    else:
        print("preprocessing needed")

    with h5py.File(train_file_path, 'r') as train_data:
        train_max_abp = train_data['abp_max'][()]
        train_min_abp = train_data['abp_min'][()]
        train_max_ple = train_data['ple_max'][()]
        train_min_ple = train_data['ple_min'][()]
        train_ple = np.array(train_data['ple'])
        train_abp_out = np.array(train_data['abp_out'])
        train_abp_level1 = np.array(train_data['abp_level1'])
        train_abp_level2 = np.array(train_data['abp_level2'])
        train_abp_level3 = np.array(train_data['abp_level3'])
        train_abp_level4 = np.array(train_data['abp_level4'])

    with h5py.File(valid_file_path, 'r') as valid_data:
        valid_max_abp = valid_data['abp_max'][()]
        valid_min_abp = valid_data['abp_min'][()]
        valid_max_ple = valid_data['ple_max'][()]
        valid_min_ple = valid_data['ple_min'][()]
        valid_ple = np.array(valid_data['ple'])
        valid_abp_out = np.array(valid_data['abp_out'])
        valid_abp_level1 = np.array(valid_data['abp_level1'])
        valid_abp_level2 = np.array(valid_data['abp_level2'])
        valid_abp_level3 = np.array(valid_data['abp_level3'])
        valid_abp_level4 = np.array(valid_data['abp_level4'])

    with h5py.File(test_file_path, 'r') as test_data:
        test_max_abp = test_data['abp_max'][()]
        test_min_abp = test_data['abp_min'][()]
        test_max_ple = test_data['ple_max'][()]
        test_min_ple = test_data['ple_min'][()]
        test_ple = np.array(test_data['ple'])
        test_abp_out = np.array(test_data['abp_out'])
        test_abp_level1 = np.array(test_data['abp_level1'])
        test_abp_level2 = np.array(test_data['abp_level2'])
        test_abp_level3 = np.array(test_data['abp_level3'])
        test_abp_level4 = np.array(test_data['abp_level4'])

    if dataset_name == 'MultiResUNet1D':
        train_dataset = MultiResUNet1DDataset(train_ple, train_abp_out, train_abp_level1, train_abp_level2,
                                              train_abp_level3, train_abp_level4, train_max_abp, train_min_abp,
                                              train_max_ple, train_min_ple)
        valid_dataset = MultiResUNet1DDataset(valid_ple, valid_abp_out, valid_abp_level1, valid_abp_level2,
                                              valid_abp_level3, valid_abp_level4, valid_max_abp, valid_min_abp,
                                              valid_max_ple, valid_min_ple)
        test_dataset = MultiResUNet1DDataset(test_ple, test_abp_out, test_abp_level1, test_abp_level2, test_abp_level3,
                                             test_abp_level4, test_max_abp, test_min_abp, test_max_ple, test_min_ple)
    else:
        train_dataset = PPG2ABPDataset(train_ple, train_abp_out, train_abp_level1, train_abp_level2, train_abp_level3,
                                       train_abp_level4, train_max_abp, train_min_abp, train_max_ple, train_min_ple)
        valid_dataset = PPG2ABPDataset(valid_ple, valid_abp_out, valid_abp_level1, valid_abp_level2, valid_abp_level3,
                                       valid_abp_level4, valid_max_abp, valid_min_abp, valid_max_ple, valid_min_ple)
        test_dataset = PPG2ABPDataset(test_ple, test_abp_out, test_abp_level1, test_abp_level2, test_abp_level3,
                                      test_abp_level4, test_max_abp, test_min_abp, test_max_ple, test_min_ple)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=test_shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=test_shuffle)

    return [train_loader, valid_loader, test_loader]
