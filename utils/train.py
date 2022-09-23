from tqdm import tqdm
import torch
import wandb
import matplotlib.pyplot as plt
from utils.funcs import plot_graph

def train_fn(epoch, model, optimizer, criterion, dataloaders, step:str = "Train" , wandb_flag:bool = True):
    #TODO : Implement multiple loss
    with tqdm(dataloaders,desc= step, total=len(dataloaders)) as tepoch:
        model.train()
        running_loss = 0.0
        for inputs, target in tepoch:
            optimizer.zero_grad()
            tepoch.set_description(step + "%d" % epoch)
            p = model(inputs)
            loss = criterion(p,target)

            if ~torch.isfinite(loss):
                continue
            loss.backward()
            running_loss += loss.item()
            optimizer.step()

            tepoch.set_postfix(loss=running_loss / tepoch.__len__())
        if wandb_flag:
            wandb.log({step + "_loss": running_loss / tepoch.__len__()})


def test_fn(epoch, model, criterion, dataloaders, step:str = "Test" , wandb_flag:bool = True, save_img:bool = True):
    #TODO : Implement multiple loss
    #TODO : Implement save model function
    with tqdm(dataloaders,desc= step, total=len(dataloaders)) as tepoch:
        model.eval()
        running_loss = 0.0

        inference_array = []
        target_array = []

        with torch.no_grad():
            for inputs, target in tepoch:
                tepoch.set_description(step + "%d" % epoch)
                p = model(inputs)
                loss = criterion(p,target)

                if ~torch.isfinite(loss):
                    continue
                running_loss += loss.item()
                if save_img:
                    inference_array.extend(p[:][0].cpu().numpy())
                    target_array.extend(target[:][0].cpu().numpy())

                tepoch.set_postfix(loss=running_loss / tepoch.__len__())
            if wandb_flag:
                wandb.log({step + "_loss": running_loss / tepoch.__len__()})
            if wandb_flag and save_img:
                plt.clf()
                plot = plot_graph(0, 300, target_array, inference_array, "original")
                plot.savefig('graph.png', figsize=(16, 4))

            # print(ppg.ppg(inference_array,30,show=False)["heart_rate"])
            if wandb_flag and save_img:
                wandb.log({
                    "Graph": [
                        wandb.Image('graph.png')
                    ]
                })