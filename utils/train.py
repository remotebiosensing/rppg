from tqdm import tqdm
import torch
import wandb
import matplotlib.pyplot as plt
from utils.funcs import plot_graph
from params import params

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.signal import welch
from ignite.handlers import FastaiLRFinder


def bpfilter64(signal, fs):
    minfq = 0.8 * 2/fs
    maxfq = 3 * 2/fs
    fir1_len = round(len(signal) / 10)
    b, a = butter(fir1_len, [minfq, maxfq], btype='bandpass')
    signal_f = filtfilt(b, a, signal)
    return signal_f

def find_lr( model, optimizer, criterion, dataloaders):
    lr_finder = FastaiLRFinder()
    lr_finder.attach(optimizer, model, criterion)
    lr_finder.range_test(dataloaders[0], val_loader=dataloaders[1], end_lr=100, num_iter=100)
    lr_finder.plot()
    lr_finder.reset()

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

            tepoch.set_postfix(loss=running_loss / tepoch.__len__())
        if wandb_flag:
            wandb.log({step + "_loss": running_loss / tepoch.__len__()},
                      step=epoch)


def test_fn(epoch, model, criterion, dataloaders, step: str = "Test", wandb_flag: bool = True, save_img: bool = True):
    # TODO : Implement multiple loss
    # TODO : Implement save model function
    with tqdm(dataloaders, desc=step, total=len(dataloaders)) as tepoch:
        model.eval()
        running_loss = 0.0

        f_array = []
        l_array = []
        c_array = []
        t_array = []
        tt_array = []

        save_flag = True
        with torch.no_grad():
            for inputs, target in tepoch:
                tepoch.set_description(step + "%d" % epoch)
                outputs = model(inputs)
                loss = criterion(outputs, target)

                if ~torch.isfinite(loss):
                    continue
                running_loss += loss.item()

                if save_img and save_flag:
                    if params.model == "TEST":
                        f = outputs[0][0].cpu().numpy()
                        l = outputs[1][0].cpu().numpy()
                        r = outputs[2][0].cpu().numpy()
                        tt = outputs[3][0].cpu().numpy()
                        t = target[0][0].cpu().numpy()
                        f_array.extend(( f - f.min())/(f.max()-f.min()))
                        l_array.extend(( l - l.min())/(l.max()-l.min()))
                        c_array.extend(( r - r.min())/(r.max()-r.min()))
                        t_array.extend(( t - t.min())/(t.max()-t.min()))
                        tt_array.extend((tt - tt.min()) / (tt.max() - tt.min()))
                        save_flag = False

                        if step == "Test":
                            t_f = bpfilter64(t, 30)
                            t_f = (t_f - np.mean(t_f)) / np.std(t_f)

                            signal_length = len(t_f)

                            f, Pg = welch(t_f[:signal_length], fs=30, nperseg=2 ** 13)
                            Frange = np.where((f > 0.7) & (f < 4))[0]
                            idxG = np.argmax(Pg[Frange])
                            HR2_1 = f[Frange][idxG] * 60
                            print(HR2_1,target[1][0])
                    else:
                        f = outputs[0].cpu().numpy()
                        t = target[0].cpu().numpy()
                        f_array.extend(( f - f.min())/(f.max()-f.min()))
                        t_array.extend((t - t.min()) / (t.max() - t.min()))

                        if step == "Test":
                            t_f = bpfilter64(t, 30)
                            t_f = (t_f - np.mean(t_f)) / np.std(t_f)

                            signal_length = len(t_f)

                            f, Pg = welch(t_f[:signal_length], fs=30, nperseg=2 ** 13)
                            Frange = np.where((f > 0.7) & (f < 4))[0]
                            idxG = np.argmax(Pg[Frange])
                            HR2_1 = f[Frange][idxG] * 60
                            print(HR2_1, target[1][0])


                tepoch.set_postfix(loss=running_loss / tepoch.__len__())
            if wandb_flag:
                wandb.log({step + "_loss": running_loss / tepoch.__len__()},
                          step=epoch)
            if  save_img and epoch%100 == 0:
                plt.clf()
                plt.rcParams["figure.figsize"] = (16, 5)

                plt.plot(range(params.time_length * tepoch.__len__()), t_array, label='target')
                plt.plot(range(params.time_length* tepoch.__len__()),f_array,label='forehead')
                if params.model == "TEST":
                    plt.plot(range(params.time_length),tt_array,label='total')
                    plt.plot(range(params.time_length),l_array,label='left')
                    plt.plot(range(params.time_length),c_array,label='right')
                plt.legend(fontsize='x-large')
                # plt.savefig("graph2.png")
                plt.show()
                # img = plt.show()

        return running_loss


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

                if  save_img and epoch==999:
                    plt.clf()
                    plt.rcParams["figure.figsize"] = (16, 5)
                    plt.plot(range(params.time_length),t_array,label='target')
                    plt.plot(range(params.time_length),f_array,label='forehead')
                    plt.plot(range(params.time_length),l_array,label='left')
                    plt.plot(range(params.time_length),c_array,label='right')
                    plt.legend(fontsize='x-large')
                    plt.savefig("graph2.png")
                    # plt.show()
                    # img = plt.show()
                    # img.savefig('graph.png', figsize=(16, 5))
                    # wandb.log({
                    #     "Graph": [
                    #         wandb.Image('graph.png')
                    #     ]
                    # })

        return running_losses
