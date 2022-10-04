import os
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import json
import wandb
import preprocessing.utils.signal_utils as su
import postprocessing.post_signal_utils as psu
import numpy as np

with open('/home/paperc/PycharmProjects/VBPNet/config/parameter.json') as f:
    json_data = json.load(f)
    param = json_data.get("parameters")
    channels = json_data.get("parameters").get("in_channels")
    sampling_rate = json_data.get("parameters").get("sampling_rate")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(125)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(125)
else:
    print("cuda not available")


def test(model, test_loader, loss_n, loss_d, loss_s, idxx):
    model.eval()
    plot_flag = True
    test_avg_cost = 0
    test_cost_sum = 0
    n_cost_sum = 0
    d_cost_sum = 0
    s_cost_sum = 0
    with tqdm(test_loader, desc='Test', total=len(test_loader), leave=True) as test_epoch:
        idx = 0
        with torch.no_grad():
            for X_test, Y_test, d, s in test_epoch:
                idx += 1
                hypothesis = torch.squeeze(model(X_test)[0])
                pred_d = torch.squeeze(model(X_test)[1])
                pred_s = torch.squeeze(model(X_test)[2])

                '''Negative Pearson Loss'''
                neg_cost = loss_n(hypothesis, Y_test)
                '''DBP Loss'''
                d_cost = loss_d(pred_d, d)
                '''SBP Loss'''
                s_cost = loss_s(pred_s, s)
                ''' Total Loss'''
                cost = neg_cost + d_cost + s_cost
                test_epoch.set_postfix(loss=cost.item())

                test_cost_sum += cost
                test_avg_cost = test_cost_sum / idx
                n_cost_sum += neg_cost
                n_avg_cost = n_cost_sum / idx
                d_cost_sum += d_cost
                d_avg_cost = d_cost_sum / idx
                s_cost_sum += s_cost
                s_avg_cost = s_cost_sum / idx
                wandb.log({"Test Loss": test_avg_cost,
                           "Test Negative Pearson Loss": n_avg_cost,
                           "Test Systolic Loss": s_avg_cost,
                           "Test Diastolic Loss": d_avg_cost}, step=idxx)
                # wandb.log({"Test Loss": cost,
                #            "Test Negative Pearson Loss": neg_cost,
                #            "Test Systolic Loss": s_cost,
                #            "Test Diastolic Loss": d_cost}, step=idxx)
                if plot_flag:
                    plot_flag = False
                    N = 0
                    h = hypothesis[N].cpu().detach()
                    y = Y_test[N].cpu().detach()
                    # plt.title("Epoch :" + str(idx + 1))
                    plt.subplot(3, 1, 1)
                    plt.plot(y)
                    plt.title("Epoch :" + str(idxx + 1) + "\nTarget ( s:" +
                              str(np.round(np.mean(su.get_systolic(y.detach().cpu().numpy())), 2)) + " / d:" +
                              str(np.round(np.mean(su.get_diastolic(y.detach().cpu().numpy())), 2)) + ")")
                    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.70)

                    plt.subplot(3, 1, 2)
                    plt.plot(psu.shape_scaler(h))
                    plt.title("Correlation")
                    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.70)
                    plt.subplot(3, 1, 3)
                    plt.plot(y)
                    plt.plot(psu.feature_combiner(h, pred_s, pred_d))
                    plt.title(
                        "Prediction ( s:" + str(np.round(pred_s.detach().cpu()[N].__float__(), 2)) + " / d:"
                        + str(np.round(pred_d.detach().cpu()[N].__float__(), 2)) + ")")
                    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.70)
                    wandb.log({"Prediction": wandb.Image(plt)})
                    plt.show()

    return test_avg_cost

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
