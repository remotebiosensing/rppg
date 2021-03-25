import model
import bvpdataset
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchsummary
import numpy as np

writer = SummaryWriter()

transform = transforms.Compose([transforms.ToTensor()])
dataset = bvpdataset.bvpdataset(
    data_path="subject_test.npz",
    transform=transform)
train_set, val_set = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8),int(len(dataset)*0.2+1)],generator=torch.Generator().manual_seed(1))
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
val_loader = DataLoader(val_set, batch_size=128, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Available devices', torch.cuda.device_count())
print('Current cuda device', torch.cuda.current_device())
print(torch.cuda.get_device_name(device))

GPU_NUM = 1
torch.cuda.set_device(GPU_NUM)
model = model.DeepPhys(in_channels=3, out_channels=32, kernel_size=3).to(device)
torchsummary.summary(model, ((3,36,36),(3,36,36)), )
MSEloss = torch.nn.MSELoss()
Adadelta = optim.Adadelta(model.parameters(), lr=1)
for epoch in range(1000000):
    print("==="+str(epoch))
    running_loss = 0.0
    for i_batch, (avg, mot, lab) in enumerate(train_loader):
        Adadelta.zero_grad()
        avg, mot, lab = avg.to(device), mot.to(device), lab.to(device)
        output = model(avg, mot)
        loss = MSEloss(output, lab)
        # if torch.isnan(loss):
        #     continue
        loss.backward()
        running_loss += loss.item()
        Adadelta.step()
        if i_batch is 0:
            writer.add_scalar('training loss', running_loss, epoch)
        # writer.add_scalar('training loss',running_loss / 128 ,epoch * len(train_loader) + i_batch)
    with torch.no_grad():
        val_loss = 0.0
        for k, (avg, mot, lab) in enumerate(val_loader):
            avg, mot, lab = avg.to(device), mot.to(device), lab.to(device)
            val_output = model(avg, mot)
            v_loss = MSEloss(val_output, lab)
            # if torch.isnan(v_loss):
            #     continue
            val_loss += v_loss
            if k is 0:
                writer.add_scalar('val loss', v_loss/128, epoch)
            # writer.add_scalar('val loss', val_loss / 128, epoch * len(val_loader) + i_batch)
    writer.close()
print('Finished Training')
