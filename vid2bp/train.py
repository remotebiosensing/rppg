from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import vid2bp.utils.train_utils as tu
from vid2bp.nets.loss.loss import SelfScaler, MAPELoss, NegPearsonLoss
import torch


def train(model, dataset, loss_list, optimizer, scheduler, epoch, scaler=True):
    model.train()
    # scale_loss = SelfScaler().to('cuda:0')
    # mape_loss = MAPELoss().to('cuda:0')
    # neg_loss = NegPearsonLoss().to('cuda:0')

    # avg_cost_list = []
    # dy_avg_cost_list = []
    # ddy_avg_cost_list = []
    # for _ in range(len(loss_list)):
    #     avg_cost_list.append(0)
    #     dy_avg_cost_list.append(0)
    #     ddy_avg_cost_list.append(0)

    with tqdm(dataset, desc='Train-{}'.format(str(epoch)), total=len(dataset),
              leave=True) as train_epoch:
        for idx, (X_train, Y_train, dy, ddy, d, s) in enumerate(train_epoch):
            optimizer.zero_grad()
            hypothesis, dbp, sbp = model(X_train)
            # dy_hypothesis = torch.diff(hypothesis, dim=1)[:, 89:269]
            # ddy_hypothesis = torch.diff(torch.diff(hypothesis, dim=1), dim=1)[:, 88:268]
            # avg_cost_list, cost = tu.calc_losses(avg_cost_list, loss_list, hypothesis, Y_train, idx + 1)
            # dy_avg_cost_list, dy_cost = tu.calc_losses(dy_avg_cost_list, loss_list, dy_hypothesis, dy, idx + 1)
            # ddy_avg_cost_list, ddy_cost = tu.calc_losses(ddy_avg_cost_list, loss_list, ddy_hypothesis, ddy, idx + 1)
            cost = loss_list[0](hypothesis, Y_train)
            # dy_cost = loss_list[0](dy_hypothesis, dy)
            # ddy_cost = loss_list[0](ddy_hypothesis, ddy)
            dbp_cost = loss_list[1](dbp, d)
            sbp_cost = loss_list[2](sbp, s)
            scale_cost = loss_list[3](dbp, sbp)

            # total_cost = cost + dy_cost + ddy_cost + dbp_cost + sbp_cost + scale_cost
            total_cost = cost + dbp_cost + sbp_cost + scale_cost

            postfix_dict = {}
            # for i in range(len(loss_list)):
            #     postfix_dict[(str(loss_list[i]))[:-2]] = (round(avg_cost_list[i], 3))

            postfix_dict['y'] = round(cost.item(), 3)
            # postfix_dict['dy'] = round(dy_cost.item(), 3)
            # postfix_dict['ddy'] = round(ddy_cost.item(), 3)
            postfix_dict['dbp'] = round(dbp_cost.item(), 3)
            postfix_dict['sbp'] = round(sbp_cost.item(), 3)
            postfix_dict['dovers'] = round(scale_cost.item(), 3)

            train_epoch.set_postfix(losses=postfix_dict, tot=total_cost.__float__())
            # (cost + dy_mape_cost + ddy_mape_cost + ple_cost).backward()
            total_cost.backward()

            optimizer.step()
        scheduler.step()

    return total_cost.__float__()
