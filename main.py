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

from dataset.dataset_loader import dataset_loader, split_data_loader
from log import log_info_time
from loss import loss_fn
from models import is_model_support, get_model, summary
from optim import optimizer
from utils.dataset_preprocess import preprocessing, dataset_split
from utils.train import train_fn, test_fn

bpm_flag = False
K_Fold_flag = False
model_save_flag = False
log_flag = False
wandb_flag = False
random_seed = 0
save_img_flag = False

# for Reproducible model
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

# Define Kfold Cross Validator
if K_Fold_flag:
    kfold = KFold(n_splits=5, shuffle=True)


now = datetime.datetime.now()
os.environ["CUDA_VISIBLE_DEVICES"] = "9"

with open('params.json') as f:
    jsonObject = json.load(f)
    __PREPROCESSING__ = jsonObject.get("__PREPROCESSING__")
    __TIME__ = jsonObject.get("__TIME__")
    __MODEL_SUMMARY__ = jsonObject.get("__MODEL_SUMMARY__")
    options = jsonObject.get("options")
    params = jsonObject.get("params")
    hyper_params = jsonObject.get("hyper_params")
    model_params = jsonObject.get("model_params")
    wandb_params = jsonObject.get("wandb")
#

if wandb_flag:
    wandb.init(project=wandb_params["project"], entity=wandb_params["entity"])

"""
TEST FOR LOAD
"""
# model = torch.load(params["model_root_path"] +model_params["name"] +params["dataset_name"] )
"""
Check Model Support
"""
is_model_support(model_params["name"], model_params["name_comment"], log_flag)
'''
Generate preprocessed data hpy file 
'''
if __PREPROCESSING__:
    if __TIME__:
        start_time = time.time()

    preprocessing(save_root_path=params["save_root_path"],
                  model_name=model_params["name"],
                  data_root_path=params["data_root_path"],
                  dataset_name=params["dataset_name"],
                  train_ratio=params["train_ratio"],
                  log_flag=log_flag)
    if __TIME__:
        log_info_time("preprocessing time \t:", datetime.timedelta(seconds=time.time() - start_time))

'''
Setting Learning Model
'''
if __TIME__:
    start_time = time.time()

# model = [get_model(model_params["name"])]
model = get_model(model_params["name"], log_flag).cuda()

if __MODEL_SUMMARY__:
    summary(model, model_params["name"], log_flag)

if __TIME__:
    log_info_time("model initialize time \t: ", datetime.timedelta(seconds=time.time() - start_time))

'''
Load dataset before using Torch DataLoader
'''
if __TIME__:
    start_time = time.time()

dataset = dataset_loader(save_root_path=params["save_root_path"],
                         model_name=model_params["name"],
                         dataset_name=params["dataset_name"],
                         option="train")

datasets = dataset_split(dataset, [0.7, 0.1, 0.2])

if __TIME__:
    log_info_time("load train hpy time \t: ", datetime.timedelta(seconds=time.time() - start_time))

if __TIME__:
    start_time = time.time()

if __TIME__:
    log_info_time("load test hpy time \t: ", datetime.timedelta(seconds=time.time() - start_time))

'''
    Call dataloader for iterate dataset
'''
if __TIME__:
    start_time = time.time()

data_loaders = split_data_loader(datasets, params["train_batch_size"], params["train_shuffle"],
                                 params["test_shuffle"])

if __TIME__:
    log_info_time("generate dataloader time \t: ", datetime.timedelta(seconds=time.time() - start_time))

'''
Setting Loss Function
'''
if __TIME__:
    start_time = time.time()
criterion = loss_fn(hyper_params["loss_fn"], log_flag)
criterion2 = loss_fn("L1", log_flag)

# if torch.cuda.is_available():
# TODO: implement parallel training
# if options["parallel_criterion"] :
#     print(options["parallel_criterion_comment"])
#     criterion = DataParallelCriterion(criterion,device_ids=[0, 1, 2])

if __TIME__:
    log_info_time("setting loss func time \t: ", datetime.timedelta(seconds=time.time() - start_time))
'''
Setting Optimizer
'''
if __TIME__:
    start_time = time.time()
# optimizer = [optimizer(mod.parameters(),hyper_params["learning_rate"], hyper_params["optimizer"]) for mod in model[0]]
# scheduler = [lr_scheduler.ExponentialLR(optim,gamma=0.99) for optim in optimizer]
optimizer = optimizer(model.parameters(), hyper_params["learning_rate"], hyper_params["optimizer"])
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

if wandb_flag:
    wandb.config = {
        "learning_rate": hyper_params["learning_rate"],
        "epochs": hyper_params["epochs"],
        "batch_size": params["train_batch_size"]
    }

if __TIME__:
    log_info_time("setting optimizer time \t: ", datetime.timedelta(seconds=time.time() - start_time))

'''
Model Training Step
'''
min_val_loss = 100.0

# dataloaders


for epoch in range(hyper_params["epochs"]):
    train_fn(epoch, model, optimizer, criterion, data_loaders[0], "Train",wandb_flag)
    if data_loaders.__len__() == 3:
        _ = test_fn(epoch, model, criterion, data_loaders[1], "Val", wandb_flag, save_img_flag )
    if epoch % 10 == 0:
        running_loss = test_fn(epoch, model, criterion, data_loaders[-1], "Test", wandb_flag, save_img_flag )
