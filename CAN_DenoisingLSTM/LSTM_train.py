# import NN
import cv2
import NN_github as NN
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torchvision
from scipy import io
from LSTM_DATA import timeseries
import datetime
from torchsummary import summary

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
mat_file = io.loadmat('./Noise_estimate/Noise_estimate_red.mat')
Xtrain = mat_file['train']
Ytrain = mat_file['test']
LSTM = NN.LSTM()
summary(LSTM, (60, 4))
LSTM.to(device)
MSEloss = torch.nn.MSELoss()
Adam = optim.Adam(LSTM.parameters(), lr=0.001)
n_epoch = 11


dataset = timeseries(Xtrain, Ytrain)
train_loader = DataLoader(dataset, batch_size=32)
for epoch in range(0, n_epoch):
    for batch_idx, (input_data, GT) in enumerate(train_loader):
        input_data, GT = input_data.to(device), GT.to(device)
