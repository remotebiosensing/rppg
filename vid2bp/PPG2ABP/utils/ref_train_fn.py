from tqdm import tqdm
import torch
import wandb


def train_fn(epoch, app_model, model, optimizer, criterion, dataloaders, step: str = "Train ", wandb_flag: bool = False):
    with tqdm(dataloaders, desc=step, total=len(dataloaders)) as tepoch:
        model.train()
        running_loss = 0.0
        for inputs, target in tepoch:
            inputs, level1, level2, level3, level4 = app_model(inputs)
            optimizer.zero_grad()
            tepoch.set_description(step + "%d" % epoch)
            pred = model(inputs)
            loss = criterion(pred, target)

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


def test_fn(epoch, app_model, model, criterion, dataloaders, step: str = "Test", wandb_flag: bool = True, save_img: bool = True):
    with tqdm(dataloaders, desc=step, total=len(dataloaders)) as tepoch:
        model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for inputs, target in tepoch:
                inputs, level1, level2, level3, level4 = app_model(inputs)
                tepoch.set_description(step + "%d" % epoch)
                pred = model(inputs)
                loss = criterion(pred, target)

                if ~torch.isfinite(loss):
                    continue
                running_loss += loss.item()

                tepoch.set_postfix(loss=running_loss / tepoch.__len__())

        return running_loss / tepoch.__len__()
