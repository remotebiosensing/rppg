import os
import sys

import torch.optim as optim
import torch
from torch.utils.data import DataLoader

from preprocessing import customdataset
from nets.modules.bvp2abp import bvp2abp
from nets.loss import loss

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('----- GPU INFO -----\nDevice:', device)  # 출력결과: cuda
print('Count of using GPUs:', torch.cuda.device_count())
print('Current cuda device:', torch.cuda.current_device(), '\n--------------------')

torch.manual_seed(125)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(125)
else:
    print("cuda not available")

# TODO TRAIN_PATIENT_RANGE 뿐만 아니라 CUSTOMDATASET에서 TRAIN_DATASET 비율 조정가능하게 변경 Done
# TODO ORDER랑 TESTORDER 하나로 정리하기 Done

''' train data load '''
# TODO use hdf5 file for training Done
dataset = "uci"
samp_rate = sampling_rate["60"]
channel = channels["sixth"]
out_channel = param['out_channels']
write_path = root_path + data_path[dataset][1]

with h5py.File(write_path + "case(" + str(channel[-1]) + ")_len(" + str(3) +
               ")_" + str(int(param["chunk_size"] / 125) * samp_rate) + "fft.hdf5", "r") as f:
    train_ple, train_abp = np.array(f['ple']), np.array(f['abp'])
    train_dataset = customdataset.CustomDataset(x_data=train_ple, y_data=train_abp)
    train_loader = DataLoader(train_dataset, batch_size=hyper_param["batch_size"], shuffle=True)

''' model train '''
''' - wandb setup '''
wandb.init(project="VBPNet", entity="paperchae")
wandb.config = wb["config"]

# model = bvp2abp(in_channels=channel[0], out_channels=param['out_channels'],
#                 kernel_size=hyper_param['kernel_size'])
model = bvp2abp(in_channels=channel[0])
if torch.cuda.is_available():
    model = model.to(device)
    loss1 = loss.NegPearsonLoss().to(device)
    loss3 = torch.nn.L1Loss(reduction='mean').to(device)
    # loss2 = loss.rmseLoss().to(device)
    # loss3 = nn.MSELoss().to(device)
    # loss2 = loss.fftLoss().to(device)
    loss2 = torch.nn.L1Loss().to(device)
else:
    print("Use Warning : Please load model on cuda! (Loaded on CPU)")
    model = model.to('cpu')
    loss1 = loss.NegPearsonLoss().to('cpu')
    loss2 = torch.nn.L1Loss().to('cpu')
    # loss3 = nn.L1Loss(reduction='mean').to('cpu')

# optimizer
optimizer = optim.AdamW(model.parameters(), lr=hyper_param["learning_rate"], weight_decay=hyper_param["weight_decay"])
# scheduler
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=hyper_param["gamma"])

print('batchN :', train_loader.__len__())

costarr = []
cnt = 0
# with tqdm(training_epochs, desc='Train',total=len(training_epochs)) as train_epochs:
for epoch in range(hyper_param["epochs"]):
    avg_cost = 0
    cost_sum = 0
    with tqdm(train_loader, desc='Train', total=len(train_loader), leave=True) as train_epochs:
        idx = 0
        for X_train, Y_train in train_epochs:
            idx += 1
            hypothesis = torch.squeeze(model(X_train))
            optimizer.zero_grad()

            '''Negative Pearson Loss'''
            cost1 = loss1(hypothesis, Y_train)
            '''RMSE Loss'''
            # cost2 = loss2(torch.squeeze(hypothesis), Y_train)
            '''MSE Loss'''
            # cost3 = loss2(hypothesis, Y_train)
            '''FFT Loss'''
            # cost2 = loss2(torch.fft.fft(hypothesis), torch.fft.fft(Y_train))
            '''MAE Loss'''
            # cost2 = loss2(pred_size, S)
            if idx == 1 and epoch % 10 == 0:
                h = hypothesis[0].cpu().detach()
                y = Y_train[0].cpu().detach()
                plt.subplot(2,1,1)
                plt.plot(y)
                plt.title("Target")
                plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.70)

                # plt.plot((h - h.min()) * (y.max() - y.min()))
                plt.subplot(2,1,2)
                plt.plot(h)
                plt.title("Prediction")
                plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.70)

                plt.show()

            '''Total Loss'''
            cost = cost1# * cost2
            cost.backward()
            optimizer.step()

            # avg_cost += cost / train_loader.__len__()
            cost_sum += cost
            avg_cost = cost_sum / idx
            train_epochs.set_postfix(loss=avg_cost.item())
            wandb.log({"Loss": cost,
                       "Negative Pearson Loss": cost1},step=epoch)
                       # "MAE Loss": cost2}, step=epoch)
                       # "FFT Loss": cost2}, step=epoch)
                       # "RMSE Loss":cost2},step=epoch)

        scheduler.step()
        costarr.append(avg_cost.__float__())

print('cost :', costarr[-1])

t_val = np.array(range(len(costarr)))
plt.plot(t_val, costarr)
plt.title('NegPearsonLoss * MAELoss')
plt.show()

# model save
torch.save(model, param["save_path"] + 'model_' + str(channel[1]) + '_NegMAE_' + dataset + '_lr_' + str(
    param["learning_rate"]) + 'fft.pt')
# torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, PATH + 'all.tar')

# TODO 1. 혈압기기와 기준기기의 차이의 정도에 따라 모델의 등급이 나뉘는 것 찾아보기
# TODO 2. PPNET 참고하여 평가지표 확인하기
# TODO 3. SAMPLING RATE 에 따른 차이 확인
# TODO 4. READRECORD() 순서는 [ABP,PLE] / 나머지 순서 다 맞추기
