import os
import sys

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from preprocessing import customdataset
from nets.modules.bvp2abp import bvp2abp
from nets.loss import loss
from train import train

import json
import wandb
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
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

''' - wandb setup '''
wandb.init(project="VBPNet", entity="paperchae")
# wandb.config = wb["config"]


def main():
    # TODO use hdf5 file for training Done
    dataset = "uci"
    samp_rate = sampling_rate["60"]
    channel = channels["sixth"]
    # out_channel = param['out_channels']
    read_path = root_path + data_path[dataset][1]

    '''train dataset load'''
    with h5py.File(read_path + "case(" + str(channel[-1]) + ")_len(" + str(3) +
                   ")_" + str(int(param["chunk_size"] / 125) * samp_rate) + "_train(True).hdf5", "r") as train_f:
        train_ple, train_abp, train_size = np.array(train_f['ple']), np.array(train_f['abp']), np.array(train_f['size'])
        train_dataset = customdataset.CustomDataset(x_data=train_ple, y_data=train_abp, size_factor=train_size)
        train_loader = DataLoader(train_dataset, batch_size=hyper_param["batch_size"], shuffle=True)

    '''test dataset load'''
    with h5py.File(read_path + "case(" + str(channel[-1]) + ")_len(" + str(1) +
                   ")_" + str(int(param["chunk_size"] / 125) * samp_rate) + "_train(False).hdf5", "r") as test_f:
        test_ple, test_abp, test_size = np.array(test_f['ple']), np.array(test_f['abp']), np.array(test_f['size'])
        test_dataset = customdataset.CustomDataset(x_data=test_ple, y_data=test_abp, size_factor=test_size)
        test_loader = DataLoader(test_dataset, batch_size=hyper_param["batch_size"], shuffle=True)
    print('batchN :', train_loader.__len__())

    '''model train'''
    model = bvp2abp(in_channels=channel[0])
    train(model=model, device=DEVICE, train_loader=train_loader,test_loader=test_loader, epochs=hyper_param["epochs"])


    # model save
    torch.save(model, param["save_path"] + 'model_' + str(channel[1]) + '_NegMAE_' + dataset + '_lr_' + str(
        hyper_param["learning_rate"]) + '_size_factor(sbp+amp).pt')
    # torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, PATH + 'all.tar')

    # TODO 1. 혈압기기와 기준기기의 차이의 정도에 따라 모델의 등급이 나뉘는 것 찾아보기
    # TODO 2. PPNET 참고하여 평가지표 확인하기
    # TODO 3. SAMPLING RATE 에 따른 차이 확인 done
    # TODO 4. READRECORD() 순서는 [ABP,PLE] / 나머지 순서 다 맞추기 done


if __name__ == "__main__":
    main()
