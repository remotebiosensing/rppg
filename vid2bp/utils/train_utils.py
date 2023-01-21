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


def get_model(model_name: str, device, stage: int = 1):
    lr, wd, ga, oc = get_model_parameter(model_name)
    if model_name == 'BPNet':
        if stage == 1:
            from vid2bp.nets.modules.bvp2abp import bvp2abp
            model = bvp2abp(in_channels=3,
                            out_channels=oc,
                            target_samp_rate=60,
                            dilation_val=2).to(device)
            # model_loss = [loss.NegPearsonLoss().to(device)]
            model_loss = [loss.NegPearsonLoss().to(device), loss.DBPLoss().to(device), loss.SBPLoss().to(device),
                          loss.ScaleLoss().to(device)]
            # model_loss = [loss.MAPELoss().to(device)]
            # model_loss = [loss.NegPearsonLoss().to(device), loss.ScaledVectorCosineSimilarity().to(device)]
            # model_loss = [loss.ScaledVectorCosineSimilarity().to(device)]
            # model_loss = [loss.SelfScaler().to(device)]
            model_optim = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            model_scheduler = optim.lr_scheduler.ExponentialLR(model_optim, gamma=ga)
            # model_scheduler = optim.lr_scheduler.LambdaLR(optimizer=model_optim, lr_lambda=ga)
        else:
            from vid2bp.nets.modules.ver2.SignalAmplificationModule import SignalAmplifier
            model = SignalAmplifier().to(device)
            model_loss = [loss.DBPLoss().to(device), loss.SBPLoss().to(device), loss.ScaleLoss().to(device)]
            model_optim = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
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
                if cnt > 10:
                    flag = False
                    break
    return flag


def calc_losses(avg_cost_list, loss_list, hypothesis, target, cnt):
    total_cost = torch.tensor(0.0).to('cuda:0')
    for idx, l in enumerate(loss_list):
        current_cost = l(hypothesis, target)
        total_cost += current_cost
        avg_cost_list.append(get_avg_cost(avg_cost_list[idx], current_cost, cnt))
        avg_cost_list[idx] = get_avg_cost(avg_cost_list[idx], current_cost, cnt)

    # cost = sum(cost_list)

    return avg_cost_list, total_cost


def get_avg_cost(avg_cost, current_cost, cnt):
    return (avg_cost * (cnt - 1) + current_cost.item()) / cnt


def model_save(train_cost_arr, val_cost_arr, model, save_point, model_name, dataset_name):
    print('\ncurrent train cost :', round(train_cost_arr[-1], 4), '/ prior_cost :', round(train_cost_arr[-2], 4),
          ' >> trained :', round(train_cost_arr[-2] - train_cost_arr[-1], 4))
    print('current val cost :', round(val_cost_arr[-1], 4), '/ prior_cost :', round(val_cost_arr[-2], 4),
          ' >> trained :', round(val_cost_arr[-2] - val_cost_arr[-1], 4))
    save_path = "/home/paperc/PycharmProjects/Pytorch_rppgs/vid2bp/weights/" + '{}_{}_{}.pth'.format(model_name,
                                                                                                     dataset_name,
                                                                                                     save_point[-1])
    print('saving model :', save_path)
    torch.save(model.state_dict(), save_path)
    try:
        prior_path = "/home/paperc/PycharmProjects/Pytorch_rppgs/vid2bp/weights/" + '{}_{}_{}.pth'.format(model_name,
                                                                                                          dataset_name,
                                                                                                          save_point[
                                                                                                              -2])
        os.remove(prior_path)
        print('removed prior model :', prior_path)
        return save_path
    except:
        print('failed to remove prior model')
        return save_path
