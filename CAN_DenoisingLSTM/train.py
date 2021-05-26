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
from bvpdataset import bvpdataset
import datetime
from torchsummary import summary

folder_name = datetime.datetime.now()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()
DeepPhys = NN.DeepPhys()
LSTM = NN.LSTM()
summary(LSTM, (60, 4))
summary(DeepPhys,((3,36,36),(3,36,36)))

# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
# DeepPhys = nn.DataParallel(DeepPhys)
DeepPhys.to(device)
MSEloss = torch.nn.MSELoss()
Adadelta = optim.Adadelta(DeepPhys.parameters(), lr=1.0)
dataset = np.load("./preprocessing/dataset/UBFC_testset_face_48.npz")
dataset = bvpdataset(A=dataset['A'], M=dataset['M'], T=dataset['T'])
train_set, val_set = torch.utils.data.random_split(dataset,
                                                   [int(np.floor(len(dataset) * 0.7)),
                                                    int(np.ceil(len(dataset) * 0.3))],
                                                   generator=torch.Generator().manual_seed(1))
# train_set, val_set = torch.utils.data.random_split(dataset,
#                                                    [int(len(dataset) * 0.8),
#                                                     int(len(dataset) * 0.2)])

train_loader = DataLoader(train_set, batch_size=32, shuffle=False)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
tmp_val_loss = 100

for epoch in range(10000):
    running_loss = 0.0
    for i, (inputs_A, inputs_M, inputs_T) in enumerate(train_loader):
        inputs_A, inputs_M, inputs_T = inputs_A.to(device), inputs_M.to(device), inputs_T.to(device)
        img_grid = torchvision.utils.make_grid(inputs_A[:5], nrow=1)
        writer.add_image('Appearance', img_grid)
        mot_grid = torchvision.utils.make_grid(inputs_M[:5], nrow=1)
        writer.add_image('Motion', mot_grid)
        output = DeepPhys(inputs_A, inputs_M)
        mask1, mask2 = output[1], output[2]
        # mask1_cpu = mask1.cpu().clone().detach()
        # if i is 10:
        #     cv2.imshow('Appearance', mask1_cpu[0].permute(1, 2, 0).numpy())
        writer.add_image('mask1', mask1[0], epoch)
        writer.add_image('mask2', mask2[0], epoch)
        loss = MSEloss(output[0], inputs_T)
        Adadelta.zero_grad()
        loss.backward()
        Adadelta.step()
        running_loss += loss.item()
        print('Train : [%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 32))
        writer.add_scalar('train_loss', running_loss, epoch)
    with torch.no_grad():
        val_loss = 0.0
        for j, (val_A, val_M, val_T) in enumerate(val_loader):
            val_A, val_M, val_T = val_A.to(device), val_M.to(device), val_T.to(device)
            val_output = DeepPhys(val_A, val_M)
            v_loss = MSEloss(val_output[0], val_T)
            val_loss += v_loss.item()
            print('Validation : [%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, val_loss / 32))
            writer.add_scalar('validation_loss', val_loss, epoch)
            if tmp_val_loss > (val_loss / 32):
                checkpoint = {'Epoch': epoch,
                              'state_dict': DeepPhys.state_dict(),
                              'optimizer': Adadelta.state_dict()}
                torch.save(checkpoint,
                           './model_checkpoint/checkpoint_' + str(folder_name.day) + "d_" + str(
                               folder_name.hour) + "h_" + str(folder_name.minute) + 'm.pth')
                tmp_val_loss = val_loss / 32
                print(tmp_val_loss)
        writer.close()
        print('Finished Training')
