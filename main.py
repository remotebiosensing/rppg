import datetime
import json
import os
import random
import time

import numpy as np
import torch
import wandb
from sklearn.model_selection import KFold
from torch.optim import lr_scheduler
#

from dataset.dataset_loader import dataset_loader, split_data_loader
from log import log_info_time
from loss import loss_fn
from models import is_model_support, get_model, summary, get_ver_model
from optim import optimizer
from utils.dataset_preprocess import preprocessing, dataset_split
from utils.train import train_fn, test_fn

from params import params

# for Reproducible model
# torch.manual_seed(random_seed)
# torch.cuda.manual_seed(random_seed)
# torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
# torch.backends.cudnn.deterministic = False
# torch.backends.cudnn.benchmark = False
# np.random.seed(random_seed)
# random.seed(random_seed)

# Define Kfold Cross Validator
if params.k_fold_flag:
    kfold = KFold(n_splits=5, shuffle=True)

now = datetime.datetime.now()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
if params.__PREPROCESSING__:
    if params.__TIME__:
        start_time = time.time()

    preprocessing()

    # save_root_path: str = "/media/hdd1/dy_dataset/",
    # model_name: str = "DeepPhys",
    # data_root_path: str = "/media/hdd1/",
    # dataset_name: str = "UBFC",
    # train_ratio: float = 0.8,
    # face_detect_algorithm: int = 0,
    # divide_flag: bool = True,
    # fixed_position: bool = True,
    # log_flag: bool = True):

    if params.__TIME__:
        log_info_time("preprocessing time \t:", datetime.timedelta(seconds=time.time() - start_time))

if torch.cuda.is_available():
    print('cuda is available')
else:
    print('cuda is not available')

'''
Setting Learning Model
'''
if params.__TIME__:
    start_time = time.time()

model = get_model()
# model = get_model(model_params["name"], log_flag).cuda()

if params.__MODEL_SUMMARY__:
    summary()

if params.__TIME__:
    log_info_time("model initialize time \t: ", datetime.timedelta(seconds=time.time() - start_time))

'''
Load dataset before using Torch DataLoader
'''
if params.__TIME__:
    start_time = time.time()

dataset = dataset_loader()

datasets = dataset_split(dataset, [params.train_ratio, params.val_ratio, params.test_ratio])

if params.__TIME__:
    log_info_time("load train hpy time \t: ", datetime.timedelta(seconds=time.time() - start_time))

if params.__TIME__:
    start_time = time.time()

if params.__TIME__:
    log_info_time("load test hpy time \t: ", datetime.timedelta(seconds=time.time() - start_time))

'''
    Call dataloader for iterate dataset
'''
if params.__TIME__:
    start_time = time.time()

data_loaders = split_data_loader(datasets, params["train_batch_size"], params["train_shuffle"],
                                 params["test_shuffle"])

if params.__TIME__:
    log_info_time("generate dataloader time \t: ", datetime.timedelta(seconds=time.time() - start_time))

'''
Setting Loss Function
'''
if params.__TIME__:
    start_time = time.time()
criterion = loss_fn()
# criterion2 = loss_fn("L1", log_flag)


# TODO: implement parallel training
# if options["parallel_criterion"] :
#     print(options["parallel_criterion_comment"])
#     criterion = DataParallelCriterion(criterion,device_ids=[0, 1, 2])

if params.__TIME__:
    log_info_time("setting loss func time \t: ", datetime.timedelta(seconds=time.time() - start_time))
'''
Setting Optimizer
'''
if params.__TIME__:
    start_time = time.time()
# optimizer = [optimizer(mod.parameters(),hyper_params["learning_rate"], hyper_params["optimizer"]) for mod in model[0]]
# scheduler = [lr_scheduler.ExponentialLR(optim,gamma=0.99) for optim in optimizer]


opt = []
sch = []
idx = 0
for ver in range(10):
    model = get_ver_model(ver).cuda()
    if params.wandb_flag:
        wandb.init(project=params.wandb_project_name, entity=params.wandb_entity,
                   name="APNET_" + str(ver) + "_" + params.loss_fn)

    opt.append(optimizer(model.parameters(), params.lr, params.optimizer))
    # optimizer = optimizer(model.parameters(), hyper_params["learning_rate"], hyper_params["optimizer"])
    # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    sch.append(lr_scheduler.ExponentialLR(opt[idx], gamma=0.99))

    if params.wandb_flag:
        wandb.config = {
            "learning_rate": params.lr,
            "epochs": params.epoch,
            "batch_size": params.train_batch_size
        }

    if params.__TIME__:
        log_info_time("setting optimizer time \t: ", datetime.timedelta(seconds=time.time() - start_time))

    '''
    Model Training Step
    '''
    min_val_loss = 100.0
    min_test_loss = 100.0
    # dataloaders

for epoch in range(params.epoch):
    train_fn(epoch, model, optimizer, criterion, data_loaders[0], "Train", params.wandb_flag)
    if data_loaders.__len__() == 3:
        val_loss = test_fn(epoch, model, criterion, data_loaders[1], "Val", params.wandb_flag, params.save_img_flag)
    if min_val_loss > val_loss:
        min_val_loss = val_loss
        running_loss = test_fn(epoch, model, criterion, data_loaders[-1], "Test", params.wandb_flag, params.save_img_flag)
        if min_test_loss > running_loss:
            min_test_loss = running_loss
            torch.save(model.state_dict(),
                       params.model_root_path + params.dataset_name + "_" + params.model + "_" + params.loss_fn)
    # if epoch % 10 == 0:

    if params.wandb_flag:
        wandb.finish()
