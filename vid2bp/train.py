from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import vid2bp.utils.train_utils as tu
from vid2bp.nets.loss.loss import SelfScaler
import torch


def train(model, dataset, loss_list, optimizer, scheduler, epoch, scaler=True):
    model.train()
    scale_loss = SelfScaler().to('cuda:0')

    avg_cost_list = []
    dy_avg_cost_list = []
    ddy_avg_cost_list = []
    for _ in range(len(loss_list)):
        avg_cost_list.append(0)
        dy_avg_cost_list.append(0)
        ddy_avg_cost_list.append(0)

    with tqdm(dataset, desc='Train{}'.format(str(epoch)), total=len(dataset),
              leave=True) as train_epoch:
        for idx, (X_train, dx, ddx, Y_train, dy, ddy, d, s) in enumerate(train_epoch):
            optimizer.zero_grad()
            hypothesis, dy_hypothesis, ddy_hypothesis, scaled_ple = model(X_train, dx, ddx, scaler=scaler)
            avg_cost_list, cost = tu.calc_losses(avg_cost_list, loss_list, hypothesis, Y_train, idx + 1)
            dy_avg_cost_list, dy_cost = tu.calc_losses(dy_avg_cost_list, loss_list, dy_hypothesis, dy, idx + 1)
            ddy_avg_cost_list, ddy_cost = tu.calc_losses(ddy_avg_cost_list, loss_list, ddy_hypothesis, ddy, idx + 1)


            ple_cost = scale_loss(scaled_ple, X_train)
            total_cost = torch.sum(torch.tensor(avg_cost_list)) + torch.sum(torch.tensor(dy_avg_cost_list)) + torch.sum(torch.tensor(ddy_avg_cost_list)) + \
                         ple_cost

            postfix_dict = {}
            for i in range(len(loss_list)):
                postfix_dict[(str(loss_list[i]))[:-2]] = (round(avg_cost_list[i], 3))
            postfix_dict['scale_variance'] = round(ple_cost.__float__(), 3)
            train_epoch.set_postfix(losses=postfix_dict, tot=total_cost)
            (cost + dy_cost + ddy_cost + ple_cost).backward()
            optimizer.step()

        scheduler.step()
        # wandb.init(project="VBPNet", entity="paperchae")
        # wandb.log({'Train Loss': total_cost}, step=epoch)
        # wandb.log({'Train Loss': train_avg_cost,
        #            'Pearson Loss': neg_cost,
        #            'STFT Loss': stft_cost}, step=epoch)
        # wandb.log({"Train Loss": cost,
        #            "Train Negative Pearson Loss": neg_cost,  # },step=epoch)
        #            "Train Systolic Loss": s_cost,
        #            "Train Diastolic Loss": d_cost}, step=epoch)
    return total_cost.__float__()
