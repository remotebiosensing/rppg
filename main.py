import os
import sys

import torch.optim as optim
from torch.utils.data import DataLoader

from preprocessing import MIMICdataset, customdataset
from preprocessing.utils import signal_utils
from nets.modules.sub_modules.bvp2abp import *
from nets.loss import loss

import json
import wandb
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import h5py

with open('parameter.json') as f:
    json_data = json.load(f)
    param = json_data.get("parameters")
    orders = json_data.get("parameters").get("in_channels")
    hyper_param = json_data.get("hyper_parameters")
    wb = json_data.get("wandb")

print(sys.version)
#
# GPU Setting
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('----- GPU INFO -----\nDevice:', device)  # 출력결과: cuda
print('Count of using GPUs:', torch.cuda.device_count())
print('Current cuda device:', torch.cuda.current_device(), '\n--------------------')

torch.manual_seed(125)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(125)

''' train data load '''
root_path = param["root_path"]
# TODO TRAIN_PATIENT_RANGE 뿐만 아니라 CUSTOMDATASET에서 TRAIN_DATASET 비율 조정가능하게 변경 Done
train_patient_range = [0, 2]
# TODO ORDER랑 TESTORDER 하나로 정리하기 Done
order = orders["third"]
'''
    order selects number of model's input channel
    orders["zero"] : use 1 channel ( f )
    orders["first"] : use 1 channel ( f' )
    orders["second"] : use 1 channel ( f'' )
    orders["third"] : use 2 channel ( f + f' )
    orders["fourth"] : use 2 channel ( f + f'' )
    orders["fifth"] : use 2 channel ( f' + f'' )
    orders["sixth"] : use 3 channel ( f + f' + f'' )
'''
preprocessed_data_path = param["preprocessed_data_path"]

train_ple, train_abp, _ = MIMICdataset.data_aggregator(root_path=root_path, degree=order[1],
                                                       train=True, percent=0.05)  # 0.05 > 2 patients
train_dataset = customdataset.CustomDataset(x_data=train_ple, y_data=train_abp)
train_loader = DataLoader(train_dataset, batch_size=hyper_param["batch_size"], shuffle=False)

'''   model train     '''
''' wandb setup '''
wandb.init(project="VBPNet", entity="paperchae")
wandb.config = wb["config"]
model = bvp2abp(in_channels=order[0], out_channels=64, kernel_size=hyper_param["kernel_size"]).to(device)

learning_rate = hyper_param["learning_rate"]
weight_decay = hyper_param["weight_decay"]
training_epochs = hyper_param["epochs"]

loss1 = loss.NegPearsonLoss().to(device)
loss2 = loss.rmseLoss().to(device)
loss4 = nn.L1Loss(reduction='mean').to(device)
# loss3 = nn.MSELoss().to(device)
# loss2 = loss.fftLoss().to(device)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

total_batch = train_loader.__len__()
print('batchN :', total_batch)

costarr = []
cnt = 0
for epoch in tqdm(range(training_epochs)):
    avg_cost = 0
    for idx, sample in enumerate(train_loader):
        X_train, Y_train = sample
        hypothesis = model(X_train)
        # print('np.shape(hypothesis) :',np.shape(hypothesis))
        optimizer.zero_grad()

        '''Negative Pearson Loss'''
        cost1 = loss1(hypothesis, Y_train)
        '''RMSE Loss'''
        # cost2 = loss2(hypothesis, Y_train)
        '''MSE Loss'''
        # cost3 = loss2(hypothesis, Y_train)
        '''L1 Loss'''
        cost4 = loss4(hypothesis, Y_train)

        '''Total Loss'''
        # if cost3 >= cost2:
        #     total_cost0 = cost1 * cost2
        # else:
        #     total_cost0 = cost1 * cost3
        cost = cost1 * cost4
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch
        wandb.log(
            {"Loss": cost,
             "Negative Pearson Loss": cost1,
             "MAE Loss": cost4}, step=epoch)
        # "RMSE Loss": cost2,
        # "MSE Loss:": cost3}, step=epoch)

    costarr.append(avg_cost.__float__())
    # if avg_cost == np.nan:
    #     print('nan idx : ', idx)
    # print('     ->     avg_cost == ', type(avg_cost.__float__()), avg_cost.__float__())
print('cost :', costarr[-1])

t_val = np.array(range(len(costarr)))
plt.plot(t_val, costarr)
plt.title('NegPearsonLoss * rmseLoss')
plt.show()

# model save
PATH = param["save_path"]
# torch.save(model, PATH + 'model_' + str(order - 1) + '_NegMAE_baseline.pt')
torch.save(model, PATH + 'model_110_NegMAE_newmodel_temp.pt')
# torch.save(model.state_dict(), PATH + 'model_state_dict.pt')
torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, PATH + 'all.tar')

# TODO 1. 혈압기기와 기준기기의 차이의 정도에 따라 모델의 등급이 나뉘는 것 찾아보기
# TODO 2. PPNET 참고하여 평가지표 확인하기
# TODO 3. SAMPLING RATE 에 따른 차이 확인
# TODO 4. READRECORD() 순서는 [ABP,PLE] / 나머지 순서 다 맞추기
