from tqdm import tqdm
import torch
import wandb
import matplotlib.pyplot as plt
from utils.funcs import plot_graph

def train_fn(epoch, model, optimizer, criterion, dataloaders, step: str = "Train ", wandb_flag: bool = False):
    with tqdm(dataloaders, desc=step, total=len(dataloaders)) as tepoch:
        model.train()
        running_loss = 0.0
        for inputs, target_out, target_level1, target_level2, target_level3, target_level4 in tepoch:
            optimizer.zero_grad()
            tepoch.set_description(step + "%d" % epoch)
            pred_out, pred_level1, pred_level2, pred_level3, pred_level4 = model(inputs)
            loss = torch.tensor(0.0).to('cuda')
            loss += criterion(pred_out, target_out)
            loss += criterion(pred_level1, target_level1) * 0.9
            loss += criterion(pred_level2, target_level2) * 0.8
            loss += criterion(pred_level3, target_level3) * 0.7
            loss += criterion(pred_level4, target_level4) * 0.6

            if ~torch.isfinite(loss):
                continue
            loss.backward()
            running_loss += loss.item()
            optimizer.step()

            tepoch.set_postfix(loss=running_loss / tepoch.__len__())
        if wandb_flag:
            wandb.log({step + "_loss": running_loss / tepoch.__len__()},
                      step=epoch)
    return running_loss / tepoch.__len__()


def test_fn(epoch, model, criterion, dataloaders, step: str = "Test", wandb_flag: bool = True, save_img: bool = True):
    with tqdm(dataloaders, desc=step, total=len(dataloaders)) as tepoch:
        model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for inputs, target_out, target_level1, target_level2, target_level3, target_level4 in tepoch:
                tepoch.set_description(step + "%d" % epoch)
                pred_out, pred_level1, pred_level2, pred_level3, pred_level4 = model(inputs)
                loss = torch.tensor(0.0).to('cuda')
                loss += criterion(pred_out, target_out)
                loss += criterion(pred_level1, target_level1) * 0.9
                loss += criterion(pred_level2, target_level2) * 0.8
                loss += criterion(pred_level3, target_level3) * 0.7
                loss += criterion(pred_level4, target_level4) * 0.6

                if ~torch.isfinite(loss):
                    continue
                running_loss += loss.item()

                tepoch.set_postfix(loss=running_loss / tepoch.__len__())

        return running_loss / tepoch.__len__()
