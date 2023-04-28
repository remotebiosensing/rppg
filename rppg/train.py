import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from tqdm import tqdm

from params import params
from rppg.utils.funcs import calculate_hr


def train_fn(epoch, model, optimizer, criterion, dataloaders, step: str = "Train ", wandb_flag: bool = True):
    # TODO : Implement multiple loss
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
            wandb.log({step + "_loss": running_loss / tepoch.__len__(),
                       },
                      step=epoch)


def test_fn(epoch, model, criterion, dataloaders, step: str = "Test", wandb_flag: bool = True, save_img: bool = True):
    # TODO : Implement multiple loss
    # TODO : Implement save model function
    with tqdm(dataloaders, desc=step, total=len(dataloaders)) as tepoch:
        model.eval()
        running_loss = 0.0

        p_array = []
        t_array = []
        hr_mae_arr = []
        save_flag = True

        with torch.no_grad():
            for inputs, target in tepoch:
                tepoch.set_description(step + "%d" % epoch)
                outputs = model(inputs)
                loss = criterion(outputs, target)

                if step == 'Test':

                    for i in range(len(outputs)):
                        out_hr = calculate_hr(outputs[i].cpu().numpy(),30)
                        target_hr = calculate_hr(target[i].cpu().numpy(),30)
                        hr_mae_arr.append(np.abs(out_hr - target_hr))


                if ~torch.isfinite(loss):
                    continue
                running_loss += loss.item()
                if save_img and save_flag:
                    p = outputs[0].cpu().numpy()
                    t = target[0].cpu().numpy()
                    p_array.extend(( p - p.min())/(p.max()-p.min()))
                    t_array.extend(( t - t.min())/(t.max()-t.min()))
                    save_flag = False
                if step == 'Test':
                    tepoch.set_postfix({'': 'loss : %.4f | HR_MAE :%.4f' % (running_loss / tepoch.__len__(),
                     np.mean(hr_mae_arr))})
                else:
                    tepoch.set_postfix({'': 'loss : %.4f |' % (running_loss / tepoch.__len__())})

            if wandb_flag:
                if step == 'Test':
                    wandb.log({step + "_loss": running_loss / tepoch.__len__(),
                               'hr_mae': np.mean(hr_mae_arr)},
                              step=epoch)
                else:
                    wandb.log({step + "_loss": running_loss / tepoch.__len__()},
                              step=epoch)
                if save_img and step=='Test' :
                    plt.clf()
                    plt.rcParams["figure.figsize"] = (16, 5)
                    plt.plot(range(params.time_length),t_array,label='target')
                    plt.plot(range(params.time_length),p_array,label='predic')
                    plt.legend(fontsize='x-large')
                    # plt.savefig("graph.png")
                    plt.show()

        return running_loss/tepoch.__len__()


def train_multi_model_fn(epoch, models, optimizers, criterion, dataloaders, step, wandb_flag):
    with tqdm(dataloaders, desc=step, total=len(dataloaders)) as tepoch:
        [model.train() for model in models]
        running_losses = [0.0 for i in range(len(models))]
        for inputs, target in tepoch:
            [optimizer.zero_grad() for optimizer in optimizers]
            tepoch.set_description(step + "%d" % epoch)
            outputs = [model(input) for input, model in zip(inputs, models)]
            losses = [criterion(outputs, target, i, epoch) for i in range(len(models))]

            for loss, running_loss, optimizer in zip(losses, running_losses, optimizers):
                if ~torch.isfinite(loss):
                    continue
                loss.requires_grad_(True)
                loss.backward()
                optimizer.step()
            running_losses = [running_loss + loss.item() for running_loss, loss in zip(running_losses, losses)]

            tepoch.set_postfix({'model' + str(i): running_loss / tepoch.__len__() for running_loss, i in
                                zip(running_losses, range(len(models)))})
        if wandb_flag:
            wandb.log({step + 'model' + str(i): running_loss / tepoch.__len__() for running_loss, i in
                       zip(running_losses, range(len(models)))}, step=epoch)


def test_multi_model_fn(epoch, models, criterion, dataloaders, step: str = "Test", wandb_flag: bool = True, save_img : bool = True):
    # TODO : Implement multiple loss
    # TODO : Implement save model function
    with tqdm(dataloaders, desc=step, total=len(dataloaders)) as tepoch:
        [model.eval() for model in models]
        running_losses = [0.0 for i in range(len(models))]

        f_array = []
        l_array = []
        c_array = []
        t_array = []
        save_flag = True
        with torch.no_grad():
            for inputs, target in tepoch:
                tepoch.set_description(step + "%d" % epoch)
                outputs = [model(input) for input, model in zip(inputs, models)]
                losses = [criterion(outputs, target, i, epoch) for i in range(len(models))]

                for loss, running_loss in zip(losses, running_losses):
                    if ~torch.isfinite(loss):
                        continue
                running_losses = [running_loss + loss.item() for running_loss, loss in zip(running_losses, losses)]

                if save_img and save_flag:
                    f = outputs[0][0].cpu().numpy()
                    l = outputs[1][0].cpu().numpy()
                    r = outputs[2][0].cpu().numpy()
                    t = target[0].cpu().numpy()
                    f_array.extend(( f - f.min())/(f.max()-f.min()))
                    l_array.extend(( l - l.min())/(l.max()-l.min()))
                    c_array.extend(( r - r.min())/(r.max()-r.min()))
                    t_array.extend(( t - t.min())/(t.max()-t.min()))
                    save_flag = False

                tepoch.set_postfix({'model' + str(i): running_loss / tepoch.__len__() for running_loss, i in
                                    zip(running_losses, range(len(models)))})
            if wandb_flag:
                wandb.log({step + 'model' + str(i): running_loss / tepoch.__len__() for running_loss, i in
                           zip(running_losses, range(len(models)))}, step=epoch)

                if  save_img :
                    plt.clf()
                    plt.rcParams["figure.figsize"] = (16, 5)
                    plt.plot(range(params.time_length),t_array,label='target')
                    plt.plot(range(params.time_length),f_array,label='forehead')
                    plt.plot(range(params.time_length),l_array,label='left')
                    plt.plot(range(params.time_length),c_array,label='right')
                    plt.legend(fontsize='x-large')
                    # plt.savefig("graph.png")
                    plt.show()
                    # img = plt.show()
                    # img.savefig('graph.png', figsize=(16, 5))
                    # wandb.log({
                    #     "Graph": [
                    #         wandb.Image('graph.png')
                    #     ]
                    # })

        return running_losses

def find_lr(model, train_loader, optimizer, criterion, init_value=1e-8, final_value=10., beta=0.98):
    num = len(train_loader)-1
    mult = (final_value / init_value) ** (1/num)
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
        avg_loss = beta * avg_loss + (1-beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta**(batch_num+1))
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

#calculate Heart Rate from PPG signal

