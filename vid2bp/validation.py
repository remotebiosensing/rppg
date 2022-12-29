import os
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import json
import wandb
import vid2bp.preprocessing.utils.signal_utils as su
import vid2bp.postprocessing.post_signal_utils as psu
import numpy as np


# with open('/home/paperc/PycharmProjects/Pytorch_rppgs/vid2bp/config/parameter.json') as f:
#     json_data = json.load(f)
#     param = json_data.get("parameters")
#     channels = json_data.get("parameters").get("in_channels")
#     sampling_rate = json_data.get("parameters").get("sampling_rate")
#
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.manual_seed(125)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(125)
# else:
#     print("cuda not available")


def validation(model, dataset, loss, epoch, scaler=True):
    model.eval()
    valid_avg_cost = 0
    valid_cost_sum = 0

    valid_epoch = tqdm(dataset, desc='Valid{}'.format(str(epoch)), total=len(dataset), leave=True)

    # with tqdm(dataset, desc='Test{}'.format(str(epoch)), total=len(dataset), leave=True) as test_epoch:
    with torch.no_grad():
        for idx, (X_val, Y_val, dia, sys, mean) in enumerate(valid_epoch):
            hypothesis = model(X_val, scaler=scaler)

            '''Negative Pearson Loss'''
            rmse_cost = loss[0](hypothesis, Y_val)
            # neg_cost = 0
            '''STFT Loss'''
            stft_cost = loss[1](hypothesis, Y_val)
            '''DBP Loss'''
            # d_cost = loss[0](pred_d, dia)
            '''SBP Loss'''
            # s_cost = loss[0](pred_s, sys)

            '''Total Loss'''
            cost = rmse_cost + stft_cost#  + d_cost + s_cost
            valid_cost_sum += cost.__float__()
            valid_avg_cost = valid_cost_sum / (idx + 1)
            valid_epoch.set_postfix(rmse=rmse_cost.__float__(), stft=stft_cost.__float__(), tot=valid_avg_cost)
        wandb.log({"Valid Loss": valid_avg_cost}, step=epoch)
        # wandb.log({"Valid Loss": valid_avg_cost,
        #            'Valid Pearson Loss': neg_cost,
        #            'STFT Loss': stft_cost}, step=epoch)
    return valid_avg_cost.__float__()
