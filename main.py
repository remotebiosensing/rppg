import datetime
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchmeta.utils.data import BatchMetaDataLoader
from tqdm import tqdm
import higher

from dataset.dataset_loader import dataset_loader
from log import log_info_time
from loss import loss_fn
from models import is_model_support, get_model, summary
from optim import optimizers
from torch.optim import lr_scheduler
from utils.dataset_preprocess import preprocessing,Meta_preprocessing
from utils.funcs import normalize, plot_graph, detrend

with open('params.json') as f:
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
Check Model Support - ok
"""
is_model_support(model_params["name"], model_params["name_comment"])
'''
Generate preprocessed data hpy file - ok
'''
if __PREPROCESSING__:
    if __TIME__:
        start_time = time.time()

    if model_params["name"] =='MetaPhys':
        Meta_preprocessing(save_root_path=params["save_root_path"],
                      model_name=model_params["name"],
                      data_root_path=params["data_root_path"],
                      dataset_name=params["dataset_name"])
    else:
        preprocessing(save_root_path=params["save_root_path"],
                      model_name=model_params["name"],
                      data_root_path=params["data_root_path"],
                      dataset_name=params["dataset_name"],
                      train_ratio=params["train_ratio"])
    if __TIME__:
        log_info_time("preprocessing time \t:", datetime.timedelta(seconds=time.time() - start_time))

'''
Load dataset before using Torch DataLoader - ok
'''
if __TIME__:
    start_time = time.time()

dataset = dataset_loader(save_root_path=params["save_root_path"],
                         model_name=model_params["name"],
                         dataset_name=params["dataset_name"],
                         option="train",

                         num_shots= meta_params["num_shots"],
                         num_test_shots = meta_params["num_test_shots"],
                         fs = meta_params["fs"],
                         unsupervised = meta_params["unsupervised"]
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
                              fs=meta_params["fs"],
                              unsupervised=meta_params["unsupervised"]
                              )
if __TIME__:
    log_info_time("load test hpy time \t: ", datetime.timedelta(seconds=time.time() - start_time))

'''
    Call dataloader for iterate dataset - ok
'''
if __TIME__:
    start_time = time.time()
if model_params["name"] == 'MetaPhys':
    train_loader = BatchMetaDataLoader(train_dataset, batch_size=params["train_batch_size"],
                                           shuffle=params["train_shuffle"])
    validation_loader = BatchMetaDataLoader(validation_dataset, batch_size=params["train_batch_size"],
                                   shuffle=params["train_shuffle"])
    test_loader = BatchMetaDataLoader(test_dataset, batch_size=params["test_batch_size"],
                                       shuffle=params["test_shuffle"])
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
Setting Learning Model - ok
'''
if __TIME__:
    start_time = time.time()
model = get_model(model_params["name"])

if model_params["name"] == "MetaPhys":
    if meta_params["pre_trained"] ==1:
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
    model.to(device)
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
inner_loss = loss_fn(meta_params["inner_loss"])
outer_loss = loss_fn(meta_params["outer_loss"])

# if torch.cuda.is_available():
# TODO: implement parallel training
# if options["parallel_criterion"] :
#     print(options["parallel_criterion_comment"])
#     criterion = DataParallelCriterion(criterion,device_ids=[0, 1, 2])

if __TIME__:
    log_info_time("setting loss func time \t: ", datetime.timedelta(seconds=time.time() - start_time))
'''
Setting Optimizer - ok
'''
if __TIME__:
    start_time = time.time()
optimizer = optimizers(model.parameters(), hyper_params["learning_rate"], hyper_params["optimizer"])
inner_optimizer = optimizers(model.parameters(), hyper_params["inner_learning_rate"], hyper_params["inner_optimizer"])
scheduler = lr_scheduler.ExponentialLR(optimizer,gamma=0.99)
if __TIME__:
    log_info_time("setting optimizer time \t: ", datetime.timedelta(seconds=time.time() - start_time))

'''

Model Training Step

min_val_loss = 100.0

for epoch in range(hyper_params["epochs"]):
    if __TIME__ and epoch == 0:
        start_time = time.time()
    with tqdm(train_loader, desc="Train ", total=len(train_loader)) as tepoch:
        model.train()
        running_loss = 0.0
        for inputs, target in tepoch:
            tepoch.set_description(f"Train Epoch {epoch}")
            print(inputs)
            outputs = model(inputs)
            loss = criterion(outputs, target)

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
                loss = criterion(outputs, target)
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

    if epoch + 1 == hyper_params["epochs"] or epoch % 10 == 0:
        if __TIME__ and epoch == 0:
            start_time = time.time()
        with tqdm(test_loader, desc="test ", total=len(test_loader)) as tepoch:
            model.eval()
            inference_array = []
            target_array = []
            with torch.no_grad():
                for inputs, target in tepoch:
                    tepoch.set_description(f"test")
                    outputs = model(inputs)
                    loss = criterion(outputs, target)
                    running_loss += loss.item()
                    tepoch.set_postfix(loss=running_loss / (params["train_batch_size"] / params["test_batch_size"]))
                    if model_params["name"] == "PhysNet":
                        inference_array.extend(normalize(outputs.cpu().numpy()[0]))
                        target_array.extend(normalize(target.cpu().numpy()[0]))
                    else:
                        inference_array.extend(outputs.cpu().numpy())
                        target_array.extend(target.cpu().numpy())
                    if tepoch.n == 0 and __TIME__:
                        save_time = time.time()

            # postprocessing
            if model_params["name"] == "DeepPhys":
                inference_array = detrend(np.cumsum(inference_array),100)
                target_array = detrend(np.cumsum(target_array),100)

            if __TIME__ and epoch == 0:
                log_info_time("inference time \t: ", datetime.timedelta(seconds=save_time - start_time))

            plot_graph(0, 300, target_array, inference_array)
'''
min_val_loss = 100.0

for epoch in range(hyper_params["epochs"]):
    if __TIME__ and epoch == 0:
        start_time = time.time()

    with tqdm(train_loader, desc="Train ", total=len(train_loader)) as tepoch:
        model.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(tepoch):
            model.zero_grad()
            tepoch.set_description(f"Train Epoch {epoch}")

            batch['train'][0] = batch['train'][0].view(params["train_batch_size"], -1, 6, 36, 36)
            batch['test'][0] = batch['test'][0].view(params["train_batch_size"], -1, 6, 36, 36)
            batch['train'][1] = batch['train'][1].view(params["train_batch_size"], -1, 1)
            batch['test'][1] = batch['test'][1].view(params["train_batch_size"], -1, 1)

            inputs, targets = batch['train']
            inputs = inputs.to(device=device)
            targets = targets.to(device=device)

            test_inputs, test_targets= batch['test']
            test_inputs = test_inputs.to(device=device)
            test_targets = test_targets.to(device=device)

            for task_idx, (input, target, test_input, test_target) in enumerate(zip(inputs, targets, test_inputs, test_targets)):
                with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=False) as (fmodel, diffopt):
                    for step in range(meta_params["num_adapt_steps"]):
                        train_logit = fmodel(input)
                        inner_loss = torch.nn.functional.mse_loss(train_logit, target)
                        diffopt.step(inner_loss)
                    test_logit = fmodel(test_input)
                    outer_loss = torch.nn.functional.mse_loss(test_logit, test_target)
                    outer_loss.backward()
            optimizer.step()
            tepoch.set_postfix(loss=outer_loss / params["train_batch_size"])
    if __TIME__ and epoch == 0:
        log_info_time("1 epoch training time \t: ", datetime.timedelta(seconds=time.time() - start_time))



    with tqdm(validation_loader, desc="Validation ", total=len(validation_loader)) as tepoch:
        model.train()
        running_loss = 0.0
        #with torch.no_grad():
        for batch_idx, batch in enumerate(tepoch):
            tepoch.set_description(f"Validation")

            batch['train'][0] = batch['train'][0].view(params["train_batch_size"], -1, 6, 36, 36)
            batch['test'][0] = batch['test'][0].view(params["train_batch_size"], -1, 6, 36, 36)
            batch['train'][1] = batch['train'][1].view(params["train_batch_size"], -1, 1)
            batch['test'][1] = batch['test'][1].view(params["train_batch_size"], -1, 1)

            inputs, targets = batch['train']

            inputs = inputs.to(device=device)
            targets = targets.to(device=device)

            test_inputs, test_targets = batch['test']

            for task_idx, (input, target, test_input, test_target) \
                    in enumerate(zip(inputs, targets, test_inputs, test_targets)):

                with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=False) as (fmodel, diffopt):
                    for step in range(meta_params["num_adapt_steps"]):
                        train_logit = fmodel(input)
                        inner_loss = torch.nn.functional.mse_loss(train_logit, target)
                        diffopt.step(inner_loss)
                    test_data_loader = DataLoader(test_input, batch_size=params["test_batch_size"], shuffle=False)
                    test_logits = torch.tensor([], device=device)
                    for i, test_batch in enumerate(test_data_loader):
                        pred = fmodel(test_batch.to(device=device)).detach()
                        test_logits = torch.cat((test_logits, pred), 0)
                    temp_test_loss = torch.nn.functional.mse_loss(test_logits, test_target.to(device=device))
                    running_loss += temp_test_loss.item()
            tepoch.set_postfix(loss=running_loss / params["test_batch_size"])

    if min_val_loss > running_loss:  # save the train model
        min_val_loss = running_loss
        checkpoint = {'Epoch': epoch,
                      'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, params["checkpoint_path"] + model_params["name"] + "/"
                   + params["dataset_name"] + "_" + str(epoch) + "_"
                   + str(min_val_loss) + '.pth')
