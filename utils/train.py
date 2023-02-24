from tqdm import tqdm
import torch
import wandb
import matplotlib.pyplot as plt
from utils.funcs import plot_graph
from params import params


def train_fn(epoch, model, optimizer, criterion, dataloaders, step: str = "Train ", wandb_flag: bool = True):
    # TODO : Implement multiple loss
    with tqdm(dataloaders, desc=step, total=len(dataloaders)) as tepoch:
        model.train()
        running_loss = 0.0
        for inputs, target in tepoch:
            optimizer.zero_grad()
            tepoch.set_description(step + "%d" % epoch)
            outputs = model(inputs)
            loss = criterion(outputs, target,epoch)

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
                loss = criterion(outputs, target, epoch)

                if ~torch.isfinite(loss):
                    continue
                running_loss += loss.item()

                if save_img and save_flag:
                    f = outputs[0][0].cpu().numpy()
                    l = outputs[1][0].cpu().numpy()
                    r = outputs[2][0].cpu().numpy()
                    tt = outputs[3][0].cpu().numpy()
                    t = target[0].cpu().numpy()
                    f_array.extend(( f - f.min())/(f.max()-f.min()))
                    l_array.extend(( l - l.min())/(l.max()-l.min()))
                    c_array.extend(( r - r.min())/(r.max()-r.min()))
                    t_array.extend(( t - t.min())/(t.max()-t.min()))
                    tt_array.extend((tt - tt.min()) / (tt.max() - tt.min()))
                    save_flag = False

                tepoch.set_postfix(loss=running_loss / tepoch.__len__())
            if wandb_flag:
                wandb.log({step + "_loss": running_loss / tepoch.__len__()},
                          step=epoch)
            if  save_img and epoch%100 == 0:
                plt.clf()
                plt.rcParams["figure.figsize"] = (16, 5)
                plt.plot(range(params.time_length), t_array, label='target')
                plt.plot(range(params.time_length),tt_array,label='total')
                plt.plot(range(params.time_length),f_array,label='forehead')
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
