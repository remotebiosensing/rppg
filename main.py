import os
import sys

import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from preprocessing import MIMICdataset, customdataset
from nets.modules.sub_modules.bvp2abp import *
from nets.loss import loss
import test

'''
wandb setup
'''
import wandb

wandb.init(project="VBPNet", entity="paperchae")

config = {
    'learning_rate': 0.01,
    'weight_decay': 0.005,
    'epochs': 50,
    'batch_size': 64
}
wandb.config = config
# torch.multiprocessing.set_start_method('spawn')


print(sys.version)

# GPU Setting
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('----- GPU INFO -----\nDevice:', device)  # 출력결과: cuda
print('Count of using GPUs:', torch.cuda.device_count())
print('Current cuda device:', torch.cuda.current_device(), '\n--------------------\n')

'''--------------------'''
root_path = '/home/paperc/PycharmProjects/BPNET/dataset/mimic-database-1.0.0/'

train_ple, train_abp = MIMICdataset.data_aggregator(root_path, slicefrom=1, sliceto=2)
'''--------------------------'''
train_dataset = customdataset.CustomDataset(x_data=train_ple, y_data=train_abp)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

''' model train '''
model = bvp2abp(in_channels=1, out_channels=64, kernel_size=3).to(device)

learning_rate = 0.01
training_epochs = 200

loss1 = loss.NegPearsonLoss().to(device)
# loss2 = nn.MSELoss().to(device)
# loss2 = loss.fftLoss().to(device)
loss2 = loss.rmseLoss().to(device)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.005)

total_batch = len(train_loader)
print('batchN :', total_batch)

costarr = []
# wandb.watch(model=model, criterion=loss, log="all", log_freq=10)
for epoch in tqdm(range(training_epochs)):
    avg_cost = 0
    for batch_idx, samples in enumerate(train_loader):
        X_train, Y_train = samples
        hypothesis = model(X_train)
        optimizer.zero_grad()

        cost1 = loss1(hypothesis, Y_train)
        cost2 = loss2(hypothesis, Y_train)
        cost = cost1 * cost2

        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch
        wandb.log({"Loss": cost, "Negative Pearson Loss": cost1, "RMSE Loss": cost2}, step=epoch)

    costarr.append(avg_cost.__float__())
    # print('     ->     avg_cost == ', avg_cost.__float__())
print('cost :', costarr[-1])

t_val = np.array(range(len(costarr)))
plt.plot(t_val, costarr)
plt.title('NegPearsonLoss * rmseLoss')
plt.show()

# model save
PATH = '/home/paperc/PycharmProjects/BPNET/weights/'
torch.save(model, PATH + 'model.pt')  # 전체 모델 저장
torch.save(model.state_dict(), PATH + 'model_state_dict.pt')  # 모델 객체의 state_dict 저장
torch.save({'model': model.state_dict(),  # 여러 가지 값 저장, 학습 중 진행 상황 저장을 위해 epoch, loss 값 등 일반 scalar 값 저장 가능
            'optimizer': optimizer.state_dict()},
           PATH + 'all.tar')

test.model_test()
