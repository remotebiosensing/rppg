import datetime
import json
import os
import random

import numpy as np
import torch
import wandb
import torch.optim.lr_scheduler as lr_scheduler

from ignite.handlers.param_scheduler import create_lr_scheduler_with_warmup

import optim
from dataset.dataset_loader import dataset_loader, split_data_loader
from log import log_info_time, time_checker
from loss import loss_fn
from models import is_model_support, get_model, summary, get_ver_model
from optim import optimizer
from utils.dataset_preprocess import preprocessing, dataset_split
from utils.train import train_fn, test_fn, train_multi_model_fn, test_multi_model_fn

from params import params

# for Reproducible model
torch.manual_seed(params.random_seed)
torch.cuda.manual_seed(params.random_seed)
torch.cuda.manual_seed_all(params.random_seed)  # if use multi-GPU
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(params.random_seed)
random.seed(params.random_seed)

os.environ["CUDA_VISIBLE_DEVICES"] = "9"

"""
TEST FOR LOAD
"""
# model = torch.load(params["model_root_path"] +model_params["name"] +params["dataset_name"] )
"""
Check Model Support
"""
is_model_support()
'''
Generate preprocessed data hpy file 
'''
time_checker("Preprocessing",preprocessing)

if torch.cuda.is_available():
    print('cuda is available')
else:
    print('cuda is not available')

'''
Setting Learning Model //Optimizer // Criterior // LR Scheduler
'''
lr_sch = {0: 10e-4, 300: 10e-5, 600: 10e-6, 900: 10e-7 }
if params.multi_model :
    models = [time_checker("get_model", get_model) for i in range(params.number_of_model)]
    opts = [optimizer(models[i].parameters(), params.lr, params.optimizer) for i in range(params.number_of_model)]
    criterion = time_checker("loss_fn", loss_fn)
    schs = [
        create_lr_scheduler_with_warmup(lr_scheduler.LambdaLR(opts[i], lr_lambda=lambda epoch: lr_sch.get(epoch, 0.998)),
                                        warmup_start_value=params.warmup_initial_lr,
                                        warmup_duration=params.warmup_iteration,
                                        warmup_end_value=params.initial_lr) for i in range(params.number_of_model)]
    min_val_loss = [100.0 for i in range(params.number_of_model)]
    min_test_loss = [100.0 for i in range(params.number_of_model)]
else:
    model = time_checker("get_model", get_model)
    opt = optimizer(model.parameters(), params.lr, params.optimizer)
    criterion = time_checker("loss_fn", loss_fn)
    sch = create_lr_scheduler_with_warmup(lr_scheduler.LambdaLR(opt, lr_lambda=lambda epoch: lr_sch.get(epoch, 0.998)),
                                        warmup_start_value=params.warmup_initial_lr,
                                        warmup_duration=params.warmup_iteration,
                                        warmup_end_value=params.initial_lr)
    min_val_loss = 100.0
    min_test_loss = 100.0

'''
Load dataset before using Torch DataLoader
'''
dataset = time_checker("load dataset", dataset_loader)
datasets = time_checker("split dataset", dataset_split, dataset = dataset, ratio = [params.train_ratio, params.val_ratio, params.test_ratio])

'''
Call dataloader for iterate dataset
'''
data_loaders = time_checker("load data", split_data_loader, datasets = datasets, batch_size=params.batch_size, train_shuffle=params.train_shuffle, test_shuffle=params.test_shuffle)


if params.wandb_flag:
    wandb.init(project=params.wandb_project_name, entity=params.wandb_entity,
               name=datetime.datetime.now().strftime('%m-%d%H:%M:%S'))
    wandb.config = {
        "learning_rate": params.lr,
        "epochs": params.epoch,
        "batch_size": params.batch_size
    }

if params.multi_model:
    for epoch in range(params.epoch):
        train_multi_model_fn(epoch,models,opts,criterion,data_loaders[0],"Train",params.wandb_flag)
        [sch(None) for sch in schs]
        if data_loaders.__len__() == 3:
            val_loss = test_multi_model_fn(epoch,models,criterion,data_loaders[1],"Val",params.wandb_flag)

        if [ m > v for m,v in zip(min_val_loss,val_loss)].count(True) > 0:
            min_val_loss = val_loss
            _ = test_multi_model_fn(epoch,models,criterion,data_loaders[-1],"Test",params.wandb_flag)


else:
    for epoch in range(params.epoch):
        train_fn(epoch, model, opt, criterion, data_loaders[0], "Train", params.wandb_flag)
        sch(None)
        if data_loaders.__len__() == 3:
            val_loss = test_fn(epoch, model, criterion, data_loaders[1], "Val", params.wandb_flag, params.save_img_flag)

        if min_val_loss > val_loss:
            min_val_loss = val_loss
            running_loss = test_fn(epoch, model, criterion, data_loaders[-1], "Test", params.wandb_flag, params.save_img_flag)
            # if min_test_loss > running_loss:
            #     min_test_loss = running_loss
            #     torch.save(model.state_dict(),
            #                params.model_root_path + params.dataset_name + "_" + params.model + "_" + params.loss_fn)
        # if epoch % 10 == 0:



if params.wandb_flag:
    wandb.finish()
