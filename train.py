import model
import bvpdataset
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import os

transform = transforms.Compose([transforms.ToTensor()])
dataset = bvpdataset.bvpdataset(
    data_path="/mnt/a7930c08-d429-42fa-a09e-15291e166a27/BVP_js/DATASET_2/M/subject_total/subject_speed_up.npz",
    transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Availabel devices', torch.cuda.device_count())
print('Current cuda device', torch.cuda.current_device())
print(torch.cuda.get_device_name(device))

GPU_NUM = 0
torch.cuda.set_device(GPU_NUM)

model = model.DeepPhys(in_channels=3, out_channels=32, kernel_size=3).to(device)
MSEloss = torch.nn.MSELoss()
Adadelta = optim.Adadelta(model.parameters(), lr=1)
for epoch in range(1000000):
    running_loss = 0.0
    for i_batch, (avg, mot, lab) in enumerate(dataloader):

        Adadelta.zero_grad()
        avg, mot, lab = avg.to(device), mot.to(device), lab.to(device)

        output = model(avg, mot)
        loss = MSEloss(output, lab)
        if torch.isnan(loss):
            continue
        loss.backward()
        #model.float()
        #torch.nn.utils.clip_grad_norm(model.parameters(),4)
        Adadelta.step()

        running_loss += loss.item()
    print('[%d] loss: %.3f [%d]' %
          (epoch + 1, running_loss, i_batch))



print('Finished Training')
