import glob
import logging
import os

import higher
import numpy as np

import torch
import torchvision
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchmeta.utils.data import BatchMetaDataLoader

from metadataset import RPPG_DATASET

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

class meta_train_model:
    def __init__(self, models, train_loader, val_loader, test_loader, criterion, lr,meta_lr, model_path, num_epochs, device):

        self.model = models
        self.num_epochs = num_epochs
        self.model_path = model_path
        self.lr = lr
        self.meta_lr = meta_lr
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.inner_optimizer = torch.optim.SGD(self.model.parameters(),lr = self.meta_lr)
        self.optimizers = torch.optim.Adadelta(models.prameters(), lr = self.lr)
        self.weights = list(models.prameters())

        self.model.to(device)
        tmp_valloss = 100

        for batch_idx,(in_avg,in_mot,in_lab),(out_avg,out_mot,out_lab) \
                in enumerate (test_loader,val_loader):
            with higher.innerloop_ctx(self.model,self.inner_optimizer, copy_initial_weights=False) as (fmodel, diffopt):
                print("Train : batch" + str(batch_idx) + "=======")
                for step in range(10): #10 = adaptation loop
                    out = fmodel((in_avg,in_mot))
                    inner_loss = F.mse_loss(out,in_lab)
                    diffopt.step(inner_loss)
                out = fmodel((out_avg,out_mot))
                outter_loss = F.mse_loss(out,out_lab)
                outter_loss.backward()
        self.inner_optimizer.step()


        for epoch in range(self.num_epochs):
            print("Train : " + str(epoch)+"=======")
            running_loss = 0.0
            for i_batch, (avg, mot, lab) in enumerate(self.train_loader):
                self.optimizers.zero_grad()
                avg, mot, lab = avg.to(device), mot.to(device), lab.to(device)

                if i_batch is 0 and epoch is 0:
                    writer.add_graph(self.model, (avg, mot))
                    images = F.interpolate(avg[:10], 128)
                    img_grid = torchvision.utils.make_grid(images, nrow=10)
                    writer.add_image('avg', img_grid)
                    images = F.interpolate(mot[:10], 128)
                    mot_grid = torchvision.utils.make_grid(images, nrow=10)
                    writer.add_image('mot', mot_grid)

                output = self.model(avg, mot)
                if i_batch is 0:
                    mask1, mask2 = self.model.appearance_model(avg)
                    writer.add_image('mask1', mask1[0], epoch)
                    writer.add_image('mask2', mask2[0], epoch)
                loss = criterion(output, lab)
                loss.backward()
                running_loss += loss.item()
                self.optimizers.step()
                if i_batch is 0:
                    writer.add_scalar('training loss', running_loss, epoch)
                # writer.add_scalar('training loss',running_loss / 128 ,epoch * len(train_loader) + i_batch)
            with torch.no_grad():
                val_loss = 0.0
                for k, (avg, mot, lab) in enumerate(val_loader):
                    avg, mot, lab = avg.to(device), mot.to(device), lab.to(device)
                    if self.model.find("TS") is not -1:
                        if avg.shape[0] % 2 is 1: # TS network need 2 images
                            continue
                    val_output = self.model(avg, mot)
                    v_loss = criterion(val_output, lab)
                    val_loss += v_loss
                    if k is 0:
                        writer.add_scalar('val loss', v_loss, epoch)
                        if tmp_valloss > val_loss:
                            checkpoint = {'Epoch': epoch,
                                          'state_dict': self.model.state_dict(),
                                          'optimizer': self.optimizers.state_dict()}
                            torch.save(checkpoint, 'checkpoint_train.pth')
                            tmp_valloss = val_loss
                    # writer.add_scalar('val loss', val_loss / 128, epoch * len(val_loader) + i_batch)
            writer.close()
        print('Finished Training')


class meta_test_model:
    def __init__(self, models, train_loader, test_loader, criterion, lr,meta_lr, model_path, num_epochs, device):

        self.model = models
        self.num_epochs = num_epochs
        self.model_path = model_path
        self.lr = lr
        self.meta_lr = meta_lr
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.inner_optimizer = torch.optim.SGD(self.model.parameters(),lr = self.meta_lr)
        self.optimizers = torch.optim.Adadelta(models.prameters(), lr = self.lr)
        self.weights = list(models.prameters())

        self.model.to(device)
        tmp_valloss = 100
        val_output = []
        for batch_idx,(in_avg,in_mot,in_lab),(out_avg,out_mot,out_lab) \
                in enumerate (test_loader,test_loader):
            with higher.innerloop_ctx(self.model,self.inner_optimizer, copy_initial_weights=False) as (fmodel, diffopt):
                print("Train : batch" + str(batch_idx) + "=======")
                for step in range(10): #10 = adaptation loop
                    out = fmodel((in_avg,in_mot))
                    inner_loss = F.mse_loss(out,in_lab)
                    diffopt.step(inner_loss)
                out = fmodel((out_avg,out_mot))
                val_output.append(out)
