import os
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import json
import wandb
import vid2bp.preprocessing.utils.signal_utils as su
import vid2bp.postprocessing.post_signal_utils as psu
import numpy as np
import vid2bp.utils.train_utils as tu
from vid2bp.nets.loss.loss import SelfScaler, MAPELoss, NegPearsonLoss


def validation(model, dataset, loss_list, epoch, scaler=True):
    model.eval()
    # scale_loss = SelfScaler().to('cuda:0')
    # mape_loss = MAPELoss().to('cuda:0')
    # neg_loss = NegPearsonLoss().to('cuda:0')

    cost_sum = 0
    # dbp_cost_sum = 0
    # sbp_cost_sum = 0
    # scale_cost_sum = 0
    amp_cost_sum = 0
    total_cost_sum = 0

    # avg_cost_list = []
    # dy_avg_cost_list = []
    # ddy_avg_cost_list = []
    # for _ in range(len(loss_list)):
    #     avg_cost_list.append(0)
    #     dy_avg_cost_list.append(0)
    #     ddy_avg_cost_list.append(0)

    with tqdm(dataset, desc='Validation-{}'.format(str(epoch)), total=len(dataset), leave=True) as valid_epoch:
        with torch.no_grad():
            for idx, (X_val, Y_val, d, s, m, info, ohe) in enumerate(valid_epoch):
                hypothesis, dbp, sbp, mbp = model(X_val, ohe)
                # dy_hypothesis = torch.diff(hypothesis, dim=1)[:, 89:269]
                # ddy_hypothesis = torch.diff(torch.diff(hypothesis, dim=-1), dim=-1)[:, 88:268]
                # avg_cost_list, cost = tu.calc_losses(avg_cost_list, loss_list, hypothesis, Y_val, idx + 1)
                # dy_avg_cost_list, dy_cost = tu.calc_losses(dy_avg_cost_list, loss_list, dy_hypothesis, dy, idx + 1)
                # ddy_avg_cost_list, ddy_cost = tu.calc_losses(ddy_avg_cost_list, loss_list, ddy_hypothesis, ddy, idx + 1)
                # dy_cost = loss_list[0](dy_hypothesis, dy)
                # ddy_cost = loss_list[0](ddy_hypothesis, ddy)
                cost = loss_list[0](hypothesis, Y_val)
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

                # dy_mape_cost = neg_loss(dhypothesis, dy)
                # ddy_mape_cost = neg_loss(ddhypothesis, ddy)
                # ple_cost = scale_loss(scaled_ple, X_test)
                # total_cost = cost + dy_cost + ddy_cost + dbp_cost + sbp_cost + scale_cost

                # total_cost = torch.sum(torch.tensor(avg_cost_list)) + ple_cost \
                #              + torch.sum(torch.tensor(dy_mape_cost)) + torch.sum(torch.tensor(ddy_mape_cost))
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

                # postfix_dict['vbp'] = round(dy_mape_cost.__float__(), 3)
                # postfix_dict['abp'] = round(ddy_mape_cost.__float__(), 3)
                # postfix_dict['scale_variance'] = round(ple_cost.__float__(), 3)
                valid_epoch.set_postfix(losses=postfix_dict, tot=total_cost.__float__())

        return total_avg_cost
