import os
import json
import numpy as np
import matplotlib.pyplot as plt

from vid2bp.nets.loss import loss

import torch
import torch.nn as nn
import torch.optim as optim


def get_model_parameter(model_name: str = 'BPNet'):
    with open('/home/paperc/PycharmProjects/Pytorch_rppgs/vid2bp/config/parameter.json') as f:
        j = json.load(f)
        params = j.get("models").get(model_name)
        out_channels = j.get("parameters").get("out_channels")
    learning_rate = params.get("learning_rate")
    weight_decay = params.get("weight_decay")
    gamma = params.get("gamma")
    return learning_rate, weight_decay, gamma, out_channels


def get_model(model_name: str, device):
    lr, wd, ga, oc = get_model_parameter(model_name)
    if model_name == 'BPNet':
        from vid2bp.nets.modules.bvp2abp import bvp2abp
        model = bvp2abp(in_channels=3, out_channels=oc).to(device)
        model_loss = [loss.NegPearsonLoss().to(device)]
        model_optim = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        model_scheduler = optim.lr_scheduler.ExponentialLR(model_optim, gamma=ga)
    elif model_name == 'Unet':
        from vid2bp.nets.modules.unet import Unet
        model = Unet()
        model_loss = [nn.MSELoss()]
        model_optim = optim.Adam(model.parameters(), lr=lr)
        model_scheduler = None  # optim.lr_scheduler.ExponentialLR(model_optim, gamma=ga)
    else:
        print('not supported model name error')
        return
    return model, model_loss, model_optim, model_scheduler


def is_learning(cost_arr):
    # print('cost :', cost_arr)
    flag = True
    if len(cost_arr) > 1:
        cost_mean = np.mean(cost_arr)
        cnt = 0
        if len(cost_arr) > 5:
            for c in reversed(cost_arr):
                if c > cost_mean:
                    cnt += 1
                if cnt > 5:
                    flag = False
                    break
    return flag


def calc_losses(avg_cost_list, losses, hypothesis, target, cnt):
    cost = torch.tensor(0.0).to('cuda:0')
    for idx, l in enumerate(losses):
        temp_loss = l(hypothesis, target)
        cost += temp_loss
        avg_cost_list[idx] = get_avg_cost(avg_cost_list[idx], temp_loss, cnt)

    # cost = sum(cost_list)

    return avg_cost_list, cost
def get_avg_cost(avg_cost, current_cost, cnt):
    return (avg_cost * (cnt - 1) + current_cost.__float__()) / cnt


def model_save(train_cost_arr, val_cost_arr, model, save_point, model_name, dataset_name):
    print('current train cost :', train_cost_arr[-1], '/ avg_cost :', train_cost_arr[-2], ' >> trained :',
          train_cost_arr[-2] - train_cost_arr[-1])
    print('current val cost :', val_cost_arr[-1], '/ avg_cost :', val_cost_arr[-2], ' >> trained :',
          val_cost_arr[-2] - val_cost_arr[-1])
    save_path = "/home/paperc/PycharmProjects/Pytorch_rppgs/vid2bp/weights/" + '{}_{}_{}.pt'.format(model_name,
                                                                                                    dataset_name,
                                                                                                    save_point[-1])
    try:
        prior_path = "/home/paperc/PycharmProjects/Pytorch_rppgs/vid2bp/weights/" + '{}_{}_{}.pt'.format(model_name,
                                                                                                         dataset_name,
                                                                                                         save_point[-2])
        os.remove(prior_path)
    except:
        torch.save(model, save_path)
