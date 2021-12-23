import copy
import datetime
import json
import time
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchmeta.utils.data import BatchMetaDataLoader
from tqdm import tqdm

from dataset.dataset_loader import dataset_loader
from log import log_info_time
from loss import loss_fn
from models import is_model_support, get_model, summary
from optim import optimizers
from torch.optim import lr_scheduler
from utils.dataset_preprocess import preprocessing
from utils.funcs import normalize, plot_graph, detrend

from nets.models.Meta import Meta

with open('meta_params.json') as f:
    jsonObject = json.load(f)
    __PREPROCESSING__ = jsonObject.get("__PREPROCESSING__")
    __TIME__ = jsonObject.get("__TIME__")
    __MODEL_SUMMARY__ = jsonObject.get("__MODEL_SUMMARY__")
    options = jsonObject.get("options")
    params = jsonObject.get("params")
    hyper_params = jsonObject.get("hyper_params")
    model_params = jsonObject.get("model_params")
    meta_params = jsonObject.get("meta_params")
#
"""
Check Model Support
"""
is_model_support(model_params["name"], model_params["name_comment"])
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
                  train_ratio=params["train_ratio"])
    if __TIME__:
        log_info_time("preprocessing time \t:", datetime.timedelta(seconds=time.time() - start_time))

'''
Load dataset before using Torch DataLoader
'''
if __TIME__:
    start_time = time.time()

dataset = dataset_loader(save_root_path=params["save_root_path"],
                         model_name=model_params["name"],
                         dataset_name=params["dataset_name"],
                         option="train",

                         num_shots=meta_params["num_shots"],
                         num_test_shots=meta_params["num_test_shots"],
                         unsupervised=meta_params["unsupervised"]
                         )

train_dataset, validation_dataset = random_split(dataset,
                                                 [int(np.floor(
                                                     len(dataset) * params["validation_ratio"])),
                                                     int(np.ceil(
                                                         len(dataset) * (1 - params["validation_ratio"])))]
                                                 )
if __TIME__:
    log_info_time("load train hpy time \t: ", datetime.timedelta(seconds=time.time() - start_time))

if __TIME__:
    start_time = time.time()
test_dataset = dataset_loader(save_root_path=params["save_root_path"],
                              model_name=model_params["name"],
                              dataset_name=params["dataset_name"],
                              option="test",

                              num_shots=meta_params["num_shots"],
                              num_test_shots=meta_params["num_test_shots"],
                              unsupervised=meta_params["unsupervised"]
                              )
if __TIME__:
    log_info_time("load test hpy time \t: ", datetime.timedelta(seconds=time.time() - start_time))

'''
    Call dataloader for iterate dataset
'''
if __TIME__:
    start_time = time.time()
if model_params["name"] == 'MetaPhys' or 'MetaPhysNet':
    train_loader = BatchMetaDataLoader(train_dataset, batch_size=params["train_batch_size"],
                                       shuffle=params["train_shuffle"])
    validation_loader = BatchMetaDataLoader(validation_dataset, batch_size=params["train_batch_size"],
                                            shuffle=params["train_shuffle"])
else:
    train_loader = DataLoader(train_dataset, batch_size=params["train_batch_size"],
                              shuffle=params["train_shuffle"])
    validation_loader = DataLoader(validation_dataset, batch_size=params["train_batch_size"],
                                   shuffle=params["train_shuffle"])
    test_loader = DataLoader(test_dataset, batch_size=params["test_batch_size"],
                             shuffle=params["test_shuffle"])
if __TIME__:
    log_info_time("generate dataloader time \t: ", datetime.timedelta(seconds=time.time() - start_time))
'''
Setting Learning Model
'''
if __TIME__:
    start_time = time.time()
model = get_model(model_params["name"])

if meta_params["pre_trained"] == 1:
    print('Using pre-trained on all ALL AFRL!')
    model.load_state_dict(torch.load('./checkpoints/AFRL_pretrained/meta_pretrained_all_AFRL.pth'))
else:
    print('Not using any pretrained models')

if torch.cuda.is_available():
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3, 4, 5, 6, 7, 8, 9'
    # TODO: implement parallel training
    # if options["parallel_criterion"] :
    #     print(options["parallel_criterion_comment"])
    #     model = DataParallelModel(model, device_ids=[0, 1, 2])
    # else:
    #     model = DataParallel(model, output_device=0)
    device = torch.device('cuda:9')
    model.to(device=device)
else:
    model = model.to('cpu')

if __MODEL_SUMMARY__:
    summary(model,model_params["name"])

if __TIME__:
    log_info_time("model initialize time \t: ", datetime.timedelta(seconds=time.time() - start_time))

'''
Setting Loss Function
'''
if __TIME__:
    start_time = time.time()
criterion = loss_fn(hyper_params["loss_fn"])
inner_criterion = loss_fn(meta_params["inner_loss"])
outer_criterion = loss_fn(meta_params["outer_loss"])
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
optimizer = optimizers(model.parameters(), hyper_params["learning_rate"], hyper_params["optimizer"])
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
if __TIME__:
    log_info_time("setting optimizer time \t: ", datetime.timedelta(seconds=time.time() - start_time))

'''
Model Training Step
'''
min_val_loss = 100.0
min_val_loss_model = None

for epoch in range(hyper_params["epochs"]):
    if __TIME__ and epoch == 0:
        start_time = time.time()
    if model_params["name"] == 'MetaPhys' or 'MetaPhysNet':
        Meta(model, train_loader, validation_loader, inner_criterion)
    else:
        with tqdm(train_loader, desc="Train ", total=len(train_loader)) as tepoch:
            model.train()
            running_loss = 0.0
            i = 0
            for inputs, target in tepoch:
                tepoch.set_description(f"Train Epoch {epoch}")
                outputs = model(inputs)

                if model_params["name"] in ["PhysNet", "PhysNet_LSTM","DeepPhys"]:
                    loss = criterion(outputs, target)
                else:
                    loss_0 = criterion(outputs[:][0], target[:][0])
                    loss_1 = criterion(outputs[:][1], target[:][1])
                    loss_2 = criterion(outputs[:][2], target[:][2])
                    loss = loss_0 + loss_2 + loss_1

                if ~torch.isfinite(loss):
                    continue

                optimizer.zero_grad()
                loss.backward()
                running_loss += loss.item()
                optimizer.step()
                tepoch.set_postfix(loss=running_loss / params["train_batch_size"])
        if __TIME__ and epoch == 0:
            log_info_time("1 epoch training time \t: ", datetime.timedelta(seconds=time.time() - start_time))

        with tqdm(validation_loader, desc="Validation ", total=len(validation_loader)) as tepoch:
            model.eval()
            running_loss = 0.0
            with torch.no_grad():
                for inputs, target in tepoch:
                    tepoch.set_description(f"Validation")
                    outputs = model(inputs)
                    if model_params["name"] in ["PhysNet", "PhysNet_LSTM", "DeepPhys"]:
                        loss = criterion(outputs, target)
                    else:
                        loss_0 = criterion(outputs[:][0], target[:][0])
                        loss_1 = criterion(outputs[:][1], target[:][1])
                        loss_2 = criterion(outputs[:][2], target[:][2])
                        loss = loss_0 + loss_2 + loss_1

                    if ~torch.isfinite(loss):
                        continue
                    running_loss += loss.item()
                    tepoch.set_postfix(loss=running_loss / params["train_batch_size"])
                if min_val_loss > running_loss:  # save the train model
                    min_val_loss = running_loss
                    checkpoint = {'Epoch': epoch,
                                  'state_dict': model.state_dict(),
                                  'optimizer': optimizer.state_dict()}
                    torch.save(checkpoint, params["checkpoint_path"] + model_params["name"] + "/"
                               + params["dataset_name"] + "_" + str(epoch) + "_"
                               + str(min_val_loss) + '.pth')
                    min_val_loss_model = copy.deepcopy(model)

        if epoch + 1 == hyper_params["epochs"] or epoch % 10 == 0:
            if __TIME__ and epoch == 0:
                start_time = time.time()
            if epoch + 1 == hyper_params["epochs"]:
                model = min_val_loss_model
            with tqdm(test_loader, desc="test ", total=len(test_loader)) as tepoch:
                model.eval()
                inference_array = []
                target_array = []
                with torch.no_grad():
                    for inputs, target in tepoch:
                        tepoch.set_description(f"test")
                        outputs = model(inputs)
                        if model_params["name"] in ["PhysNet", "PhysNet_LSTM", "DeepPhys"]:
                            loss = criterion(outputs, target)
                        else:
                            loss_0 = criterion(outputs[:][0], target[:][0])
                            loss_1 = criterion(outputs[:][1], target[:][1])
                            loss_2 = criterion(outputs[:][2], target[:][2])
                            loss = loss_0 + loss_2 + loss_1

                        if ~torch.isfinite(loss):
                            continue
                        running_loss += loss.item()
                        tepoch.set_postfix(loss=running_loss / (params["train_batch_size"] / params["test_batch_size"]))
                        if model_params["name"] in ["PhysNet","PhysNet_LSTM"]:
                            inference_array.extend(normalize(outputs.cpu().numpy()[0]))
                            target_array.extend(normalize(target.cpu().numpy()[0]))
                        else:
                            inference_array.extend(outputs[:][0].cpu().numpy())
                            target_array.extend(target[:][0].cpu().numpy())
                        if tepoch.n == 0 and __TIME__:
                            save_time = time.time()

                # postprocessing
                if model_params["name"] in ["DeepPhys"]:
                    inference_array = detrend(np.cumsum(inference_array),100)
                    target_array = detrend(np.cumsum(target_array),100)

                if __TIME__ and epoch == 0:
                    log_info_time("inference time \t: ", datetime.timedelta(seconds=save_time - start_time))

                plot_graph(0, 300, target_array, inference_array)