from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import vid2bp.utils.train_utils as tu
from vid2bp.nets.loss.loss import SelfScaler, MAPELoss, NegPearsonLoss
import torch


def train(model, dataset, loss_list, optimizer, scheduler, epoch, scaler=True):
    model.train()
    cost_sum = 0
    dbp_cost_sum = 0
    sbp_cost_sum = 0
    scale_cost_sum = 0
    amp_cost_sum = 0
    total_cost_sum = 0

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
        for idx, (X_train, Y_train, d, s, m, info, ohe) in enumerate(train_epoch):
            optimizer.zero_grad()
            hypothesis, dbp, sbp, mbp = model(X_train, ohe)
            # dy_hypothesis = torch.diff(hypothesis, dim=1)[:, 89:269]
            # ddy_hypothesis = torch.diff(torch.diff(hypothesis, dim=1), dim=1)[:, 88:268]
            # avg_cost_list, cost = tu.calc_losses(avg_cost_list, loss_list, hypothesis, Y_train, idx + 1)
            # dy_avg_cost_list, dy_cost = tu.calc_losses(dy_avg_cost_list, loss_list, dy_hypothesis, dy, idx + 1)
            # ddy_avg_cost_list, ddy_cost = tu.calc_losses(ddy_avg_cost_list, loss_list, ddy_hypothesis, ddy, idx + 1)
            cost = loss_list[0](hypothesis, Y_train)
            # dbp_cost = loss_list[1](dbp, d)
            # sbp_cost = loss_list[2](sbp, s)
            # scale_cost = loss_list[3](dbp, sbp)
            amp_cost = loss_list[-1](dbp, sbp, mbp, d, s, m)
            # total_cost = cost + dbp_cost + sbp_cost + scale_cost
            total_cost = cost + amp_cost# + scale_cost

            cost_sum += cost.item()
            avg_cost = cost_sum / (idx + 1)
            # dbp_cost_sum += dbp_cost.item()
            # dbp_avg_cost = dbp_cost_sum / (idx + 1)
            # sbp_cost_sum += sbp_cost.item()
            # sbp_avg_cost = sbp_cost_sum / (idx + 1)
            # scale_cost_sum += scale_cost.item()
            # scale_avg_cost = scale_cost_sum / (idx + 1)
            amp_cost_sum += amp_cost.item()
            amp_avg_cost = amp_cost_sum / (idx + 1)
            total_cost_sum += total_cost.item()
            total_avg_cost = total_cost_sum / (idx + 1)

            # dy_cost = loss_list[0](dy_hypothesis, dy)
            # ddy_cost = loss_list[0](ddy_hypothesis, ddy)
            # total_cost = cost + dy_cost + ddy_cost + dbp_cost + sbp_cost + scale_cost

            postfix_dict = {}
            # for i in range(len(loss_list)):
            #     postfix_dict[(str(loss_list[i]))[:-2]] = (round(avg_cost_list[i], 3))

            postfix_dict['y'] = round(avg_cost, 3)
            # postfix_dict['dy'] = round(dy_cost.item(), 3)
            # postfix_dict['ddy'] = round(ddy_cost.item(), 3)
            # postfix_dict['dbp'] = round(dbp_avg_cost, 3)
            # postfix_dict['sbp'] = round(sbp_avg_cost, 3)
            postfix_dict['amp'] = round(amp_avg_cost, 3)
            # postfix_dict['dovers'] = round(scale_avg_cost, 3)
            postfix_dict['total'] = round(total_avg_cost, 3)

            train_epoch.set_postfix(losses=postfix_dict)
            # (cost + dy_mape_cost + ddy_mape_cost + ple_cost).backward()
            total_cost.backward()

            optimizer.step()
        scheduler.step()

    return total_avg_cost
