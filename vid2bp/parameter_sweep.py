import os
import sys

import torch.optim as optim
from torch.utils.data import DataLoader

from preprocessing import customdataset
from nets.modules.sub_modules.Trend_module import *
from nets.loss import loss
import preprocessing.utils.train_utils as tu

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

print(sys.version)

# CUDA Setting
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
ch = ["zero", "third", "sixth"]
learning_rate = [0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001]
# learning_rate = [0.003, 0.002, 0.001]
weight_decay = [0.005, 0.0045, 0.004, 0.0035, 0.003, 0.0025, 0.002, 0.0015, 0.001, 0.0005]
# weight_decay = [0.0015, 0.001, 0.0005]
for c in ch:
    for (l, w) in zip(learning_rate, weight_decay):
        cnt = 0
        print("channel :", c, "learning_rate :", l, "weight_decay :", w)
        channel = channels[c]
        write_path = root_path + data_path[dataset][1]

        with h5py.File(write_path + "case(" + str(channel[-1]) + ")_len(" + str(1) +
                       ")_" + str(int(param["chunk_size"] / 125) * samp_rate) + "ns.hdf5", "r") as f:
            train_ple, train_abp = np.array(f['ple']), np.array(f['abp'])
            train_dataset = customdataset.CustomDataset(x_data=train_ple, y_data=train_abp)
            train_loader = DataLoader(train_dataset, batch_size=hyper_param["batch_size"], shuffle=False)

        ''' model train '''
        ''' - wandb setup '''
        wandb.init(project="VBPNet", entity="paperchae")
        wandb.config = wb["config"]

        model = bvp2abp(in_channels=channel[0], out_channels=param['out_channels'],
                        kernel_size=hyper_param['kernel_size'])
        if torch.cuda.is_available():
            model = model.to(device)
            loss1 = loss.NegPearsonLoss().to(device)
            loss2 = nn.L1Loss(reduction='mean').to(device)
            # loss2 = loss.rmseLoss().to(device)
            # loss3 = nn.MSELoss().to(device)
            # loss2 = loss.fftLoss().to(device)
        else:
            print("Use Warning : Please load model on cuda! (Loaded on CPU)")
            model = model.to('cpu')
            loss1 = loss.NegPearsonLoss().to('cpu')
            loss2 = nn.L1Loss(reduction='mean').to('cpu')

        # optimizer
        optimizer = optim.AdamW(model.parameters(), lr=l,
                                weight_decay=w)
        # scheduler
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=hyper_param["gamma"])

        print('batchN :', train_loader.__len__())

        costarr = []

        # TODO change tqdm to print training loss every epoch

        for epoch in tqdm(range(hyper_param["epochs"])):
            # with tqdm(training_epochs, desc='Train',total=len(training_epochs)) as train_epochs:
            avg_cost = 0
            cost_sum = 0

            costlist = []
            for idx, sample in enumerate(train_loader):
                X_train, Y_train = sample
                hypothesis = torch.squeeze(model(X_train))
                optimizer.zero_grad()

                '''Negative Pearson Loss'''
                cost1 = loss1(hypothesis, Y_train)
                '''RMSE Loss'''
                # cost2 = loss2(torch.squeeze(hypothesis), Y_train)
                '''MSE Loss'''
                # cost3 = loss2(hypothesis, Y_train)
                '''MAE Loss'''
                cost2 = loss2(hypothesis, Y_train)

                '''Total Loss'''
                cost = cost1 * cost2
                # costlist.append(cost.__float__())
                cost.backward()
                optimizer.step()

                # avg_cost += cost / train_loader.__len__()
                cost_sum += cost
                avg_cost = cost_sum / idx
                wandb.log({"Loss": cost,
                           "Negative Pearson Loss": cost1,
                           # "RMSE Loss":cost2},step=epoch)
                           "MAE Loss": cost2}, step=epoch)
            scheduler.step()
            costarr.append(avg_cost.__float__())
            # print(cost.__float__())
            if cost.__float__() > 5:
                cnt += 1
                print('cnt :', cnt, 'cost :', cost.__float__())
            if cost.__float__() == np.NaN:
                cnt += 31
                print('cnt :', cnt, 'cost :', cost.__float__())
            if cnt > 30:
                print("model is not converging.. moving on to next model")
                break

        print('cost :', costarr[-1])

        # t_val = np.array(range(len(costarr)))
        # plt.plot(t_val, costarr)
        # plt.title('NegPearsonLoss * MAELoss')
        # plt.show()

        # model save
        # print(param["save_path"] + 'model_' + str(channel[1]) + '_NegMAE' + dataset + '_lr_' + str(l) + '.pt')
        torch.save(model,
                   param["save_path"] + 'NS_model_' + str(channel[1]) + '_NegMAE_' + dataset + '_lr_' + str(l) + '.pt')
        # torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, param["save_path"] + 'all.tar')

# TODO 1. 혈압기기와 기준기기의 차이의 정도에 따라 모델의 등급이 나뉘는 것 찾아보기
# TODO 2. PPNET 참고하여 평가지표 확인하기
# TODO 3. SAMPLING RATE 에 따른 차이 확인
# TODO 4. READRECORD() 순서는 [ABP,PLE] / 나머지 순서 다 맞추기
