import os
import sys

import torch.optim as optim
from torch.utils.data import DataLoader

from preprocessing import MIMICdataset, customdataset
from nets.modules.sub_modules.bvp2abp import *
from nets.loss import loss

import json
import wandb
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

with open('parameter.json') as f:
    json_data = json.load(f)
    param = json_data.get("parameters")
    orders = json_data.get("parameters").get("in_channels")
    hyper_param = json_data.get("hyper_parameters")
    wb = json_data.get("wandb")

'''
wandb setup
'''
wandb.init(project="VBPNet", entity="paperchae")
wandb.config = wb["config"]

print(sys.version)
#
# GPU Setting
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('----- GPU INFO -----\nDevice:', device)  # 출력결과: cuda
print('Count of using GPUs:', torch.cuda.device_count())
print('Current cuda device:', torch.cuda.current_device(), '\n--------------------')


''' train data load '''
root_path = param["root_path"]
slice_range = [0, 2]
order = orders["second"]
'''
    order selects number of model's input channel
    orders["zero"] : use 1 channel ( f )
    orders["first"] : use 2 channel ( f + f' )
    orders["second"] : use 3 channel ( f + f' + f'' )
'''
if order == 1:  # return f
    train_ple, train_abp = MIMICdataset.data_aggregator(
        root_path=root_path, degree=order - 1, slicefrom=slice_range[0], sliceto=slice_range[1])
    train_dataset = customdataset.CustomDataset(x_data=train_ple, y_data=train_abp)
    train_loader = DataLoader(train_dataset, batch_size=hyper_param["batch_size"], shuffle=False)

elif order == 2:  # return np.concatenate((f, f'), axis=1)
    train_ple, train_abp = MIMICdataset.data_aggregator(
        root_path=root_path, degree=order - 1, slicefrom=slice_range[0], sliceto=slice_range[1])
    train_dataset = customdataset.CustomDataset(x_data=train_ple, y_data=train_abp, degree=order - 1)
    train_loader = DataLoader(train_dataset, batch_size=hyper_param["batch_size"], shuffle=False)


else:  # return np.concatenate((f, f', f''), axis=1)
    train_ple, train_abp = MIMICdataset.data_aggregator(
        root_path=root_path, degree=order - 1, slicefrom=slice_range[0], sliceto=slice_range[1])
    train_dataset = customdataset.CustomDataset(x_data=train_ple, y_data=train_abp, degree=order - 1)
    train_loader = DataLoader(train_dataset, batch_size=hyper_param["batch_size"], shuffle=False)
#
# else:
#     print('not supported order : please check parameter.json')
# '''--------------------------'''

''' model train '''
model = bvp2abp(in_channels=orders["second"], out_channels=16, kernel_size=3).to(device)

learning_rate = hyper_param["learning_rate"]
weight_decay = hyper_param["weight_decay"]
training_epochs = hyper_param["epochs"]

loss1 = loss.NegPearsonLoss().to(device)
loss2 = loss.rmseLoss().to(device)
# loss3 = nn.MSELoss().to(device)
# loss2 = loss.fftLoss().to(device)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

total_batch = len(train_loader)
print('batchN :', total_batch)

costarr = []
a = 2
b = 4
c = 4
su = a+b+c
ap = a/su
bp = b/su
cp = c/su
for epoch in tqdm(range(training_epochs)):
    avg_cost = 0
    for idx, samples in enumerate(train_loader):
        X_train, Y_train = samples

        hypothesis = model(X_train)
        # print('np.shape(hypothesis) :',np.shape(hypothesis))
        optimizer.zero_grad()

        '''Negative Pearson Loss'''
        cost1 = loss1(hypothesis, Y_train)*ap
        '''RMSE Loss'''
        cost2 = loss2(hypothesis, Y_train)*bp
        '''MSE Loss'''
        cost3 = loss2(hypothesis, Y_train)*cp

        '''Total Loss'''
        if cost3 >= cost2:
            total_cost0 = cost1 * cost2
        else:
            total_cost0 = cost1 * cost3

        cost = total_cost0
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch
        wandb.log(
            {"Loss": cost,
             "Negative Pearson Loss": cost1,
             "RMSE Loss": cost2,
             "MSE Loss:": cost3}, step=epoch)

    costarr.append(avg_cost.__float__())
    print('     ->     avg_cost == ', avg_cost.__float__())
print('cost :', costarr[-1])

t_val = np.array(range(len(costarr)))
plt.plot(t_val, costarr)
plt.title('NegPearsonLoss * rmseLoss')
plt.show()

# model save
PATH = param["save_path"]
torch.save(model, PATH + 'model_derivative_test.pt')
# torch.save(model.state_dict(), PATH + 'model_state_dict.pt')
torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, PATH + 'all.tar')
