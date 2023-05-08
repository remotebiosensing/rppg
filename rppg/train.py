import math
import torch
import wandb
from tqdm import tqdm
from rppg.utils.funcs import (get_hr,MAE,RMSE,MAPE,corr)
import numpy as np

def train_fn(epoch, model, optimizer, criterion, dataloaders, wandb_flag: bool = True):
    # TODO : Implement multiple loss
    step = "Train"
    with tqdm(dataloaders, desc=step, total=len(dataloaders)) as tepoch:
        model.train()
        running_loss = 0.0

        for inputs, target in tepoch:
            optimizer.zero_grad()
            tepoch.set_description(step + "%d" % epoch)
            outputs = model(inputs)
            loss = criterion(outputs, target)

            if ~torch.isfinite(loss):
                continue
            loss.requires_grad_(True)
            loss.backward()
            running_loss += loss.item()

            optimizer.step()

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
            for inputs, target in tepoch:
                tepoch.set_description(step + "%d" % epoch)
                outputs = model(inputs)
                loss = criterion(outputs, target)

                if ~torch.isfinite(loss):
                    continue
                running_loss += loss.item()
                tepoch.set_postfix({'': 'loss : %.4f |' % (running_loss / tepoch.__len__())})

            if wandb_flag:
                wandb.log({step + "_loss": running_loss / tepoch.__len__()},
                          step=epoch)

        return running_loss / tepoch.__len__()


def test_fn(epoch, model, dataloaders, model_name, cal_type,  metrics, wandb_flag: bool = True):
    # To evaluate a model by subject, you can use the meta option
    step = "Test"
    if epoch is None:
        epoch = 0
    if model_name in ["DeepPhys", "MTTS"]:
        model_type = 'DIFF'
    else:
        model_type = 'CONT'

    with tqdm(dataloaders, desc=step, total=len(dataloaders)) as tepoch:
        model.eval()
        hr_preds = []
        hr_targets = []
        with torch.no_grad():
            for inputs, target in tepoch:
                tepoch.set_description(step + "%d" % epoch)
                outputs = model(inputs)
                if model_name in ['ContrastPhys']:
                    outputs = outputs[:,-1,:]
                hr_pred, hr_target = get_hr(outputs.detach().cpu().numpy(),target.detach().cpu().numpy(),model_type= model_type ,cal_type=cal_type)
                hr_preds.extend(hr_pred)
                hr_targets.extend(hr_target)
            hr_preds = np.asarray(hr_preds)
            hr_targets = np.asarray(hr_targets)

            if "MAE" in metrics:
                print("MAE",MAE(hr_preds,hr_targets))
            if "RMSE" in metrics:
                print("RMSE",RMSE(hr_preds,hr_targets))
            if "MAPE" in metrics:
                print("MAPE",MAPE(hr_preds,hr_targets))
            if "Pearson" in metrics:
                print("Pearson",corr(hr_preds,hr_targets))




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
