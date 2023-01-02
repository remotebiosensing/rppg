from tqdm import tqdm
import torch
import wandb
import matplotlib.pyplot as plt
from utils.funcs import plot_graph
import numpy as np


def train_fn(epoch, model, optimizer, criterion, dataloaders, step: str = "Train ", wandb_flag: bool = False):
    with tqdm(dataloaders, desc=step, total=len(dataloaders)) as tepoch:
        model.train()
        running_loss = 0.0
        for inputs, target in tepoch:
            optimizer.zero_grad()
            tepoch.set_description(step + "%d" % epoch)
            pred = model(inputs) * dataloaders.dataset.std + dataloaders.dataset.mean

            loss = criterion(pred, target)

            if ~torch.isfinite(loss):
                continue
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
            tepoch.set_postfix(loss=running_loss / tepoch.__len__())

        if wandb_flag:
            plt.plot(pred[0].cpu().detach().numpy(), label="pred")
            plt.plot(target[0].cpu().detach().numpy(), label="target")
            plt.title("epoch " + str(epoch + 1))
            plt.legend()
            wandb.log({"train pred and target": wandb.Image(plt),
                       step + "_loss": running_loss / tepoch.__len__()},
                      step=epoch)
            plt.clf()
    return running_loss / tepoch.__len__()


def test_fn(epoch, model, criterion, dataloaders, step: str = "Test", wandb_flag: bool = True, save_img: bool = True):
    with tqdm(dataloaders, desc=step, total=len(dataloaders)) as tepoch:
        model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for inputs, target in tepoch:
                tepoch.set_description(step + "%d" % epoch)
                pred = model(inputs) * dataloaders.dataset.std + dataloaders.dataset.mean
                loss = criterion(pred, target)
                if ~torch.isfinite(loss):
                    continue
                running_loss += loss.item()

                tepoch.set_postfix(loss=running_loss / tepoch.__len__())

        if wandb_flag:
            plt.plot(pred[0].cpu().detach().numpy(), label="pred")
            plt.plot(target[0].cpu().detach().numpy(), label="target")
            plt.title("epoch " + str(epoch + 1))
            plt.legend()
            wandb.log({step + " pred and target": wandb.Image(plt),
                       step + "_loss": running_loss / tepoch.__len__()},
                      step=epoch)
            plt.clf()
        return running_loss / tepoch.__len__()
