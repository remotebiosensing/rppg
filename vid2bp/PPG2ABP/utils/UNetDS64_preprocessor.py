from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import multiprocessing
import os

import h5py
from torch.utils.data import random_split

length = 352


def prepareData(model, X_train, X_val, X_test):
    """
    Prepares data for 2nd stage training

    Arguments:
        model {pytorch_model} -- pytorch model
        X_train {array} -- X train
        X_val {array} -- X val
        X_test {array} -- X test
        Y_train {array} -- Y train
        Y_val {array} -- Y val
        Y_test {array} -- Y test

    Returns:
        tuple -- tuple of X_train, X_val and X_test for 2nd stage training
    """

    X2_train = []

    X2_val = []

    X2_test = []

    YPs = model.predict(X_train)

    YPs = YPs.detach().cpu().numpy()

    for i in tqdm(range(len(X_train))):
        X2_train.append(np.array(YPs[i]))

    YPs = model.predict(X_val)

    YPs = YPs.detach().cpu().numpy()

    for i in tqdm(range(len(X_val))):
        X2_val.append(np.array(YPs[i]))

    YPs = model.predict(X_test)

    YPs = YPs.detach().cpu().numpy()

    for i in tqdm(range(len(X_test))):
        X2_test.append(np.array(YPs[i]))

    X2_train = torch.Tensor(np.array(X2_train))

    X2_val = torch.Tensor(np.array(X2_val))

    X2_test = torch.Tensor(np.array(X2_test))

    return (X2_train, X2_val, X2_test)


def prepareDataDS(model, X):
    """
    Prepares data for 2nd stage training in the deep supervised pipeline

    Arguments:
        model -- pytorch model
        X {array} -- array being X train or X val

    Returns:
        X {array} -- suitable X for 2nd stage training
    """

    X2 = []

    YPs = model(X)

    for i in tqdm(range(len(X)), desc='Preparing Data for DS'):
        X2.append(np.array(YPs[0][i]))

    X2 = torch.Tensor(np.array(X2))

    return X2


def prepareLabel(Y):
    def approximate(inp, w_len):
        op = []
        for i in range(0, len(inp), w_len):
            op.append(np.mean(inp[i:i + w_len]))

        return np.array(op)

    out = {}
    out['out'] = []
    out['level1'] = []
    out['level2'] = []
    out['level3'] = []
    out['level4'] = []

    for y in tqdm(Y, desc='Preparing Label for DS'):
        cA1 = approximate(np.array(y).reshape(length), 2)
        cA2 = approximate(np.array(y).reshape(length), 4)
        cA3 = approximate(np.array(y).reshape(length), 8)
        cA4 = approximate(np.array(y).reshape(length), 16)

        out['out'].append(np.array(y.reshape(length, 1)))
        out['level1'].append(np.array(cA1.reshape(length // 2, 1)))
        out['level2'].append(np.array(cA2.reshape(length // 4, 1)))
        out['level3'].append(np.array(cA3.reshape(length // 8, 1)))
        out['level4'].append(np.array(cA4.reshape(length // 16, 1)))

    out['out'] = np.array(out['out'])  # converting to numpy array
    out['level1'] = np.array(out['level1'])
    out['level2'] = np.array(out['level2'])
    out['level3'] = np.array(out['level3'])
    out['level4'] = np.array(out['level4'])

    return out


def preprocessing(original_data_path, save_path, length, mode):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    return_dict['abp'] = manager.dict()
    return_dict['abp']['out'] = manager.list()
    return_dict['abp']['level1'] = manager.list()
    return_dict['abp']['level2'] = manager.list()
    return_dict['abp']['level3'] = manager.list()
    return_dict['abp']['level4'] = manager.list()
    return_dict['ple'] = manager.list()

    data_file = original_data_path + mode + ".hdf5"
    data_file = h5py.File(data_file, 'r')

    ple_max = np.max(data_file['ple'][:, 0])
    ple_min = np.min(data_file['ple'][:, 0])
    abp_max = np.max(data_file['abp'])
    abp_min = np.min(data_file['abp'])
    process = []
    # multiprocessing
    num_cpu = multiprocessing.cpu_count()
    loop = int(len(data_file['abp']) / num_cpu)
    for i in range(num_cpu):
        if i == num_cpu - 1:
            p = multiprocessing.Process(target=preprocess_approximate_Dataset,
                                        args=(data_file['abp'][i * loop:],
                                              data_file['ple'][i * loop:, 0], length, return_dict))
        else:
            p = multiprocessing.Process(target=preprocess_approximate_Dataset,
                                        args=(data_file['abp'][i * loop:(i + 1) * loop],
                                              data_file['ple'][i * loop:(i + 1) * loop, 0], length, return_dict))

        p.start()
        process.append(p)

    for p in process:
        p.join()
    data_file_path = save_path + mode + ".hdf5"
    data_file = h5py.File(data_file_path, "w")

    data_file.create_dataset('ple', data=np.array(return_dict['ple']))
    data_file.create_dataset('abp_out', data=np.array(return_dict['abp']['out']))
    data_file.create_dataset('abp_level1', data=np.array(return_dict['abp']['level1']))
    data_file.create_dataset('abp_level2', data=np.array(return_dict['abp']['level2']))
    data_file.create_dataset('abp_level3', data=np.array(return_dict['abp']['level3']))
    data_file.create_dataset('abp_level4', data=np.array(return_dict['abp']['level4']))
    data_file.create_dataset('ple_max', data=np.array(ple_max))
    data_file.create_dataset('ple_min', data=np.array(ple_min))
    data_file.create_dataset('abp_max', data=np.array(abp_max))
    data_file.create_dataset('abp_min', data=np.array(abp_min))
    data_file.close()


def preprocess_approximate_Dataset(ABP, PLE, length, return_dict, max_abp, min_abp, max_ple, min_ple):
    ABP = (ABP[:, :length]-min_abp) / (max_abp-min_abp)
    PLE = (PLE[:]-min_ple) / (max_ple-min_ple)

    ABP = prepareLabel(ABP)
    return_dict['abp']['out'].extend(ABP['out'])
    return_dict['abp']['level1'].extend(ABP['level1'])
    return_dict['abp']['level2'].extend(ABP['level2'])
    return_dict['abp']['level3'].extend(ABP['level3'])
    return_dict['abp']['level4'].extend(ABP['level4'])
    return_dict['ple'].extend(PLE)


if __name__ == '__main__':
    original_data_path = "/home/najy/PycharmProjects/vid2bp_datasets/"
    save_path = "/home/najy/PycharmProjects/PPG2ABP_datasets/"

    for mode in ['train', 'val', 'test']:
        preprocessing(original_data_path, save_path, 352, mode)
