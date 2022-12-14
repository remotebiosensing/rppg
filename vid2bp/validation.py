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


def validation(model_n, model, validation_loader, loss):
    model.eval()
    val_avg_cost = 0
    val_cost_sum = 0

    if model_n == 'Unet':
        loss_m = loss
        with tqdm(validation_loader, desc='Validation{}', total=len(validation_loader), leave=True) as validation_epoch:
            idx = 0
            with torch.no_grad():
                for X_val, Y_val in validation_epoch:
                    idx += 1
                    hypothesis = torch.squeeze(model(X_val))
                    cost = loss_m(hypothesis, Y_val)
                    validation_epoch.set_postfix(loss=cost.item())
                    val_cost_sum += cost
                    val_avg_cost = val_cost_sum / idx

    elif model_n == 'BPNet':
        # loss_n = loss[0]
        # loss_d = loss[1]
        # loss_s = loss[2]
        loss_fft = loss[0]

        with tqdm(validation_loader, desc='Validation', total=len(validation_loader), leave=True) as validation_epoch:
            idx = 0
            with torch.no_grad():
                for X_val, Y_val, d, s in validation_epoch:
                    idx += 1
                    hypothesis = torch.squeeze(model(X_val))
                    # pred_d = torch.squeeze(model(X_val)[1])
                    # pred_s = torch.squeeze(model(X_val)[2])

                    # '''Negative Pearson Loss'''
                    # neg_cost = loss_n(hypothesis, Y_val)
                    # '''DBP Loss'''
                    # d_cost = loss_d(pred_d, d)
                    # '''SBP Loss'''
                    # s_cost = loss_s(pred_s, s)
                    ''' Total Loss'''
                    # cost = neg_cost + d_cost + s_cost
                    cost =loss_fft(hypothesis, Y_val)

                    if not np.isnan(cost.__float__()):
                        val_cost_sum += cost.__float__()
                        val_avg_cost = val_cost_sum / idx
                        # validation_epoch.set_postfix(_=val_avg_cost, r=neg_cost.item(), d=d_cost.item(), s=s_cost.item())
                        validation_epoch.set_postfix(_=val_avg_cost)
                    else:
                        print('nan error')
                        continue


    return val_avg_cost

# import h5py
# from torch.utils.data import DataLoader
# from preprocessing import customdataset
# from nets.loss import loss
#
# # GPU Setting
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print('----- GPU INFO -----\nDevice:', DEVICE)  # 출력결과: cuda
# print('Count of using GPUs:', torch.cuda.device_count())
# print('Current cuda device:', torch.cuda.current_device(), '\n--------------------')
#
# torch.manual_seed(125)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(125)
# else:
#     print("cuda not available")
#
# with open('config/parameter.json') as f:
#     json_data = json.load(f)
#     param = json_data.get("parameters")
#     channels = json_data.get("parameters").get("in_channels")
#     hyper_param = json_data.get("hyper_parameters")
#     root_path = json_data.get("parameters").get("root_path")  # mimic uci containing folder
#     data_path = json_data.get("parameters").get("dataset_path")  # raw mimic uci selection
#     sampling_rate = json_data.get("parameters").get("sampling_rate")
#
# dataset = "uci"
# channel = channels["third"]
# samp_rate = sampling_rate['60']
# read_path = root_path + data_path[dataset][1]
# model = torch.load('/home/paperc/PycharmProjects/VBPNet/weights/model_uci_(f+f\'+f\'\')_corr07_ppgscaled.pt')
# model = model.to(DEVICE)
# loss_neg = loss.NegPearsonLoss().to(DEVICE)
# loss_d = loss.dbpLoss().to(DEVICE)
# loss_s = loss.sbpLoss().to(DEVICE)
# '''test dataset load'''
# with h5py.File('/home/paperc/PycharmProjects/VBPNet/dataset/BPNet_uci/AHAclass/case(f+f\'+f\'\')_len(1)_360_train(False)_std(3)_3(stage1).hdf5',
#                "r") as test_f:
#     print("<test dataset>")
#     test_ple, test_abp, test_size = np.array(test_f['ple']), np.array(test_f['abp']), np.array(test_f['size'])
#     test_dataset = customdataset.CustomDataset(x_data=test_ple, y_data=test_abp, size_factor=test_size)
#     test_loader = DataLoader(test_dataset, batch_size=hyper_param["batch_size"], shuffle=True)
#
#
# test(model, test_loader, loss_neg, loss_d, loss_s, 0)
