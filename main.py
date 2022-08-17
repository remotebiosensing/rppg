import os
import sys

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
# import pytorch_model_summary
import numpy as np

from preprocessing import MIMICdataset, customdataset
from preprocessing.utils import math_module
from nets.modules.sub_modules.bvp2abp import *
from nets.loss import loss

# import wandb
#
# wandb.init()
# config = {
#     'epochs': 500,
#     'batch_size': 7500,
#     'learning_rate': 0.0001,
#     'weight_decay': 0.00005,
#     'seed': 42
# }
print(sys.version)

# GPU Setting
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)  # 출력결과: cuda
print('Count of using GPUs:', torch.cuda.device_count())
print('Current cuda device:', torch.cuda.current_device())

root_path = '/home/paperc/PycharmProjects/BPNET/dataset/mimic-database-1.0.0/'
# root_path = '/tmp/pycharm_project_811/VBPNet/dataset/mimic-database-1.0.0/'
t_path1 = '039/03900004'
t_path2 = '039/03900001'
train_path = root_path + t_path1
test_path = root_path + t_path1
chunk_size = 10
# print('data_path:', data_path)

# read record
train_record = MIMICdataset.read_record(train_path)
test_record = MIMICdataset.read_record(train_path)
# returns name of file
record_name = train_record.record_name
# returns list of signals [abp, bvp]
sig_name = train_record.sig_name
# returns numpy array of signals
train_signal = train_record.p_signal
test_signal = train_record.p_signal

train_abp, train_ple = MIMICdataset.sig_slice(signals=train_signal, size=chunk_size)
test_abp, test_ple = MIMICdataset.sig_slice(signals=test_signal, size=chunk_size)

# print(abp.shape)
# print('--------------------------------')
# print(ple.shape)
# print('--------------------------------')
print('__main__ -> ', np.shape(train_abp[0]), np.shape(train_ple[0]))
# test_list = MIMICdataset.find_person(root_path)
# print(test_list)
#
# t = test_list[0]
#
# test_path = root_path + t
# print(test_path)
# new_dataset_path = '/tmp/pycharm_project_811/VBPNet/dataset/bvp2abp/'
# for l in test_list:
#     os.makedirs(new_dataset_path+l)

# train_dataset = customdataset.CustomDataset(x_data=ple[0:3], y_data=abp[0:3])
train_dataset = customdataset.CustomDataset(x_data=train_ple, y_data=train_abp)
train_loader = DataLoader(train_dataset, batch_size=7500, shuffle=False)
test_dataset = customdataset.CustomDataset(x_data=test_ple[0], y_data=test_abp[0])
test_loader = DataLoader(test_dataset, batch_size=7500, shuffle=False)

print('__main__ -> train_dataset[0][0] :', train_dataset[0][0].size())

dataiter = iter(train_loader)
seq, labels = dataiter.next()
print('__main__ -> seq_size :', seq.size(), 'label_size', labels.size())
print('train ple :', seq)
print('abp label :', labels)

# derivativetest = math_module.derivative(labels)

''' model test '''
model = bvp2abp(in_channels=1, out_channels=64, kernel_size=3).to(device)

learning_rate = 0.0001
training_epochs = 500
# loss = nn.MSELoss().to(device)
loss1 = loss.NegPearsonLoss().to(device)
# loss2 = loss.rmseLoss().to(device)
loss2 = nn.MSELoss().to(device)
# loss2 = loss.fftLoss().to(device)
# print(loss1.shape)
# print(loss2.shape)
# loss = loss1 + loss2
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.00005)

total_batch = len(train_loader)
print('batchN :', total_batch)
costarr = []
# wandb.watch(model=model, criterion=loss, log="all", log_freq=10)
for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in train_loader:
        hypothesis = model(X)

        optimizer.zero_grad()
        cost1 = loss1(hypothesis, Y)
        cost2 = loss2(hypothesis, Y)
        cost = cost1 * cost2
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch
    costarr.append(avg_cost.__float__())
    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

print(costarr)

# print(pytorch_model_summary.summary(model=model, show_input=False))

from matplotlib import pyplot as plt
import numpy as np

t_val = np.array(range(len(costarr)))
plt.plot(t_val, costarr)
plt.title('NegPearsonLoss * fftLoss')
plt.show()
