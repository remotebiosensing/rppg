import time
import datetime
import json
import numpy as np
import torch
import os


from parallel import DataParallelModel, DataParallelCriterion
from nets.Models import Deepphys
from loss import loss_fn
from optim import optimizer
from dataset.dataset_loader import dataset_loader
from torch.utils.data import DataLoader, random_split
from torch.nn import DataParallel
from utils.dataset_preprocess import preprocessing

with open('params.json') as f:
    jsonObject = json.load(f)
    __TIME__ = jsonObject.get("__TIME__")
    options = jsonObject.get("options")
    params = jsonObject.get("params")
    hyper_params = jsonObject.get("hyper_params")

'''
Generate preprocessed data hpy file 
'''
if False:
    if __TIME__:
        start_time = time.time()
    preprocessing(save_root_path=params["save_root_path"],
                  data_root_path=params["data_root_path"],
                  dataset_name=params["dataset_name"],
                  train_ratio=params["train_ratio"])
    if __TIME__:
        print("preprocessing time \t: ", datetime.timedelta(seconds=time.time() - start_time))

'''
Load dataset before using Torch DataLoader
'''
if __TIME__:
    start_time = time.time()

dataset = dataset_loader(save_root_path=params["save_root_path"],
                         dataset_name=params["dataset_name"],
                         option="train")

train_dataset, validation_dataset = random_split(dataset,
                                                 [int(np.floor(
                                                     len(dataset) * params["validation_ratio"])),
                                                     int(np.ceil(
                                                         len(dataset) * (1 - params["validation_ratio"])))]
                                                 )
if __TIME__:
    print("load train hpy time \t: ", datetime.timedelta(seconds=time.time() - start_time))

if __TIME__:
    start_time = time.time()
test_dataset = dataset_loader(save_root_path=params["save_root_path"],
                              dataset_name=params["dataset_name"],
                              option="test")
if __TIME__:
    print("load test hpy time \t: ", datetime.timedelta(seconds=time.time() - start_time))

'''
    Call dataloader for iterate dataset
'''
if __TIME__:
    start_time = time.time()
train_loader = DataLoader(train_dataset, batch_size=params["train_batch_size"],
                          shuffle=params["train_shuffle"])
validation_loader = DataLoader(validation_dataset, batch_size=params["train_batch_size"],
                               shuffle=params["train_shuffle"])
test_loader = DataLoader(test_dataset, batch_size=params["test_batch_size"],
                         shuffle=params["test_shuffle"])
if __TIME__:
    print("generate dataloader time \t: ", datetime.timedelta(seconds=time.time() - start_time))
'''
Setting Learning Model
'''
if __TIME__:
    start_time = time.time()
model = Deepphys()
if torch.cuda.is_available():
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3, 4, 5, 6, 7, 8, 9'
    # TODO: implement parallel training
    # if options["parallel_criterion"] :
    #     print(options["parallel_criterion_comment"])
    #     model = DataParallelModel(model, device_ids=[0, 1, 2])
    # else:
    #     model = DataParallel(model, output_device=0)
    model.cuda()
else:
    model = model.to('cpu')
if __TIME__:
    print("model initialize time \t: ", datetime.timedelta(seconds=time.time() - start_time))

'''
Setting Loss Function
'''
if __TIME__:
    start_time = time.time()
criterion = loss_fn(hyper_params["loss_fn"])
if criterion is None:
    print("use implemented loss functions")
    print(hyper_params["loss_fn_comment"])
    raise NotImplementedError("implement a custom function(%s) in loss.py" % hyper_params["loss_fn"])
# if torch.cuda.is_available():
    # TODO: implement parallel training
    # if options["parallel_criterion"] :
    #     print(options["parallel_criterion_comment"])
    #     criterion = DataParallelCriterion(criterion,device_ids=[0, 1, 2])

if __TIME__:
    print("setting loss func time \t: ", datetime.timedelta(seconds=time.time() - start_time))
'''
Setting Optimizer
'''
if __TIME__:
    start_time = time.time()
optimizer = optimizer(model.parameters(), hyper_params["learning_rate"],hyper_params["optimizer"])
if criterion is None:
    print("use implemented optimizer")
    print(hyper_params["optimizer_comment"])
    raise NotImplementedError("implement a custom optimizer(%s) in optimizer.py" % hyper_params["optimizer"])
if __TIME__:
    print("setting optimizer time \t: ", datetime.timedelta(seconds=time.time() - start_time))

for i,(appearance_data, motion_data, target) in enumerate(train_loader):
    print(i)
    outputs = model(appearance_data,motion_data)
    loss = criterion(outputs,target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()