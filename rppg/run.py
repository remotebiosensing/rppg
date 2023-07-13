import math
import torch
import wandb
from tqdm import tqdm
from rppg.utils.funcs import (get_hr, MAE, RMSE, MAPE, corr, IrrelevantPowerRatio)
import numpy as np
import os


def run(model, sweep, optimizer, lr_sch, criterion, cfg, dataloaders):
    log = True
    best_loss = 100000
    val_loss = 0
    eval_flag = False
    save_dir = cfg.model_save_path + cfg.fit.model + "/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    test_result = []
    if cfg.fit.train_flag:
        for epoch in range(cfg.fit.train.epochs):
            train_fn(epoch, model, optimizer, lr_sch, criterion, dataloaders[0], cfg.wandb.flag)
            val_loss = val_fn(epoch, model, criterion, dataloaders[1], cfg.wandb.flag)
            if best_loss > val_loss:
                best_loss = val_loss
                eval_flag = True
                if cfg.fit.model_save_flag:
                    torch.save(model.state_dict(), save_dir +
                               "train" + cfg.fit.train.dataset +
                               "_test" + cfg.fit.test.dataset +
                               "_imgsize" + str(cfg.fit.img_size) +
                               ".pt")
            if log:
                if cfg.fit.eval_flag and (eval_flag or (epoch + 1) % cfg.fit.eval_interval == 0):
                    test_fn(epoch, model, dataloaders[2], cfg.fit.model, cal_type=cfg.fit.test.cal_type,
                            metrics=cfg.fit.test.metric, eval_time_length=10,
                            wandb_flag=cfg.wandb.flag)
                eval_flag = False
        if not sweep:
            test_result = test_fn(0, model, dataloaders[2], cfg.fit.model, cal_type=cfg.fit.test.cal_type,
                                  metrics=cfg.fit.test.metric, eval_time_length=10,
                                  wandb_flag=cfg.wandb.flag)
        else:
            for et in cfg.fit.test.eval_time_length:
                test_result.append(test_fn(0, model, dataloaders[2], cfg.fit.model, cal_type=cfg.fit.test.cal_type,
                                           metrics=cfg.fit.test.metric, eval_time_length=et,
                                           wandb_flag=cfg.wandb.flag))
    else:
        # model = torch.load()
        if not sweep:
            test_result.append(test_fn(0, model, dataloaders[0], cfg.fit.model, cal_type=cfg.fit.test.cal_type,
                                       metrics=cfg.fit.test.metric, eval_time_length=cfg.fit.test.eval_time_length,
                                       wandb_flag=cfg.wandb.flag))
        else:
            for et in cfg.fit.test.eval_time_length:
                test_result.append(test_fn(0, model, dataloaders[0], cfg.fit.model, cal_type=cfg.fit.test.cal_type,
                                           metrics=cfg.fit.test.metric, eval_time_length=et,
                                           wandb_flag=cfg.wandb.flag))

    return test_result


def train_fn(epoch, model, optimizer, lr_sch, criterion, dataloaders, wandb_flag: bool = True):
    # TODO : Implement multiple loss
    step = "Train"

    with tqdm(dataloaders, desc=step, total=len(dataloaders)) as tepoch:
        model.train()
        running_loss = 0.0

        for te in tepoch:
            if model.__module__.split('.')[-1] == 'PhysFormer':
                inputs, target, hr = te
            else:
                inputs, target = te
            optimizer.zero_grad()
            tepoch.set_description(step + "%d" % epoch)
            outputs = model(inputs)
            if model.__module__.split('.')[-1] == 'PhysFormer':
                loss = criterion(epoch, outputs, target, hr)
            else:
                loss = criterion(outputs, target)

            if ~torch.isfinite(loss):
                continue
            loss.requires_grad_(True)
            loss.backward(retain_graph=True)
            running_loss += loss.item()

            optimizer.step()
            if lr_sch is not None:
                lr_sch.step()

            tepoch.set_postfix({'': 'loss : %.4f | ' % (running_loss / tepoch.__len__())})

        if wandb_flag:
            wandb.log({"Train" + "_loss": running_loss / tepoch.__len__(),
                       },
                      step=epoch)


def val_fn(epoch, model, criterion, dataloaders, wandb_flag: bool = True):
    # TODO : Implement multiple loss
    # TODO : Implement save model function
    step = "Val"

    with tqdm(dataloaders, desc=step, total=len(dataloaders)) as tepoch:
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for te in tepoch:
                if model.__module__.split('.')[-1] == 'PhysFormer':
                    inputs, target, hr = te
                else:
                    inputs, target = te
                tepoch.set_description(step + "%d" % epoch)
                outputs = model(inputs)
                if model.__module__.split('.')[-1] == 'PhysFormer':

                    loss = criterion(epoch, outputs, target, hr)
                else:
                    loss = criterion(outputs, target)
                if ~torch.isfinite(loss):
                    continue
                running_loss += loss.item()
                tepoch.set_postfix({'': 'loss : %.4f |' % (running_loss / tepoch.__len__())})

            if wandb_flag:
                wandb.log({step + "_loss": running_loss / tepoch.__len__()},
                          step=epoch)

        return running_loss / tepoch.__len__()

'''
D = scipy.sparse.spdiags(298, 300)
H + (Lambda ** 2) * np.dot(D.T, D) (300, 300)
'''

def test_fn(epoch, model, dataloaders, model_name, cal_type, metrics, eval_time_length, wandb_flag: bool = True):
    # To evaluate a model by subject, you can use the meta option
    step = "Test"

    if model_name in ["DeepPhys", "TSCAN", "MTTS", "BigSmall", "EfficientPhys"]:
        model_type = 'DIFF'
    else:
        model_type = 'CONT'

    model.eval()

    p = []
    t = []

    fs = 30
    time = eval_time_length

    interval = fs * time
    empty_tensor = torch.empty(1).to('cuda')
    empty_tensor2 = torch.empty(1).to('cuda')
    with tqdm(dataloaders, desc=step, total=len(dataloaders), disable=False) as tepoch:
        _pred = []
        _target = []
        with torch.no_grad():
            for te in tepoch:
                torch.cuda.empty_cache()
                if model.__module__.split('.')[-1] == 'PhysFormer':
                    inputs, target, _ = te
                else:
                    inputs, target = te
                outputs = model(inputs).detach().clone()
                empty_tensor = torch.cat((empty_tensor, outputs.squeeze()), dim=-1)
                empty_tensor2 = torch.cat((empty_tensor2, target.squeeze()), dim=-1)

    hr_pred, hr_target = get_hr(torch.stack(list(torch.split(empty_tensor[1:].detach(), 300))[:-1], dim=0),
                                torch.stack(list(torch.split(empty_tensor2[1:].detach(), 300))[:-1], dim=0),
                                model_type=model_type, cal_type=cal_type)

    hr_pred = np.asarray(hr_pred)
    hr_target = np.asarray(hr_target)

    test_result = []
    if "MAE" in metrics:
        test_result.append(round(MAE(hr_pred, hr_target), 3))
        print("MAE", MAE(hr_pred, hr_target))
    if "RMSE" in metrics:
        test_result.append(round(RMSE(hr_pred, hr_target), 3))
        print("RMSE", RMSE(hr_pred, hr_target))
    if "MAPE" in metrics:
        test_result.append(round(MAPE(hr_pred, hr_target), 3))
        print("MAPE", MAPE(hr_pred, hr_target))
    if "Pearson" in metrics:
        test_result.append(round(corr(hr_pred, hr_target)[0][1], 3))
        print("Pearson", corr(hr_pred, hr_target))
    return test_result


def find_lr(model, train_loader, optimizer, criterion, init_value=1e-8, final_value=10., beta=0.98):
    num = len(train_loader) - 1
    mult = (final_value / init_value) ** (1 / num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    losses = []
    log_lrs = []
    for batch_num, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # loss = loss[0]*10ㅌ+ loss[1]*5# + loss[2] #+ loss[3]
        # Compute the smoothed loss
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** (batch_num + 1))
        # Record the best loss
        if smoothed_loss < best_loss or batch_num == 0:
            best_loss = smoothed_loss
        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            break
        # Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        loss.requires_grad_(True)
        # Do the backward pass and update the weights
        loss.backward()
        optimizer.step()
        # Update the learning rate
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
    return log_lrs, losses

# calculate Heart Rate from PPG signal
