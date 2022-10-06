import os
import sys

import torch
from torch.utils.data import DataLoader

from preprocessing import customdataset
from nets.modules.bvp2abp import bvp2abp
from nets.modules.unet import Unet
from train import train

import json
import wandb
import numpy as np
import h5py

with open('config/parameter.json') as f:
    json_data = json.load(f)
    param = json_data.get("parameters")
    channels = json_data.get("parameters").get("in_channels")
    hyper_param = json_data.get("hyper_parameters")
    wb = json_data.get("wandb")
    root_path = json_data.get("parameters").get("root_path")  # mimic uci containing folder
    data_path = json_data.get("parameters").get("dataset_path")  # raw mimic uci selection
    sampling_rate = json_data.get("parameters").get("sampling_rate")
    models = json_data.get("parameters").get("models")

print(sys.version)

# GPU Setting
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('----- GPU INFO -----\nDevice:', DEVICE)  # 출력결과: cuda
print('Count of using GPUs:', torch.cuda.device_count())
print('Current cuda device:', torch.cuda.current_device(), '\n--------------------')

torch.manual_seed(125)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(125)
else:
    print("cuda not available")


def main(model_name, dataset_name):
    # TODO use hdf5 file for training Done
    if model_name is 'VBPNet':
        samp_rate = sampling_rate["60"]
        channel = channels["sixth"]
        read_path = root_path + data_path[dataset_name][1]
        model = bvp2abp(in_channels=channel[0])

        '''train dataset load'''
        train_filename = read_path + 'shuffled/case(' + str(channel[-1]) + ')_' + \
                         str(int(param['chunk_size'] / 125) * samp_rate) + "_train.hdf5"
        with h5py.File(train_filename, "r") as train_f:
            print(train_filename)
            train_ple, train_abp, train_size = np.array(train_f['ple']), np.array(train_f['abp']), np.array(
                train_f['size'])
            train_dataset = customdataset.CustomDataset(x_data=train_ple, y_data=train_abp, size_factor=train_size)
            train_loader = DataLoader(train_dataset, batch_size=hyper_param["batch_size"], shuffle=True)

        '''test dataset load'''
        test_filename = read_path + 'shuffled/case(' + str(channel[-1]) + ')_' + \
                        str(int(param['chunk_size'] / 125) * samp_rate) + '_test.hdf5'
        with h5py.File(test_filename, "r") as test_f:
            print(test_filename)
            test_ple, test_abp, test_size = np.array(test_f['ple']), np.array(test_f['abp']), np.array(test_f['size'])
            test_dataset = customdataset.CustomDataset(x_data=test_ple, y_data=test_abp, size_factor=test_size)
            test_loader = DataLoader(test_dataset, batch_size=hyper_param["batch_size"], shuffle=True)

    elif model_name is 'Unet':
        model = Unet()
    else:
        raise ValueError("** model name is not correct, please check supported model name in parameter.json **")
    '''model train'''
    train(model=model, device=DEVICE, train_loader=train_loader, test_loader=test_loader, epochs=hyper_param["epochs"])


if __name__ == '__main__':
    main(model_name="Unet", dataset_name="uci_unet")
