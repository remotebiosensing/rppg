import torch
import NN_github as NN
from math import sqrt
from scipy import stats
from scipy import io
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import numpy as np
from bvpdataset import bvpdataset
from inference_preprocess import detrend
from torch.utils.tensorboard import SummaryWriter
import sys

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

#
# mat_file = io.loadmat('./mask/UBFC_train_mask_0524.mat')
# Attention_mask1 = mat_file['mask1']
# Attention_mask2 = mat_file['mask2']
# Appearance = mat_file['Appearance']
# Pulse_estimate = mat_file['Pulse_estimate']

print(sys.version)
print(torch.__version__)
print('Availabel devices', torch.cuda.device_count())
print('Current cuda device', torch.cuda.current_device())
print(torch.cuda.get_device_name(device))

writer = SummaryWriter()
DeepPhys = NN.DeepPhys()

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    DeepPhys = nn.DataParallel(DeepPhys)
DeepPhys = NN.DeepPhys().to(device)
checkpoint = torch.load("./model_checkpoint/checkpoint_18d_10h_44m.pth")
DeepPhys.load_state_dict(checkpoint['state_dict'])
DeepPhys.eval()

testset = ["COHFACE_test_4"]
# testset = ["test_48"]
MAE = []
MSE = []
rMSE = []

for i in range(0, 1):
    dataset = np.load('./preprocessing/dataset/UBFC_trainset_face.npz')
    dataset = bvpdataset(A=dataset['A'], M=dataset['M'], T=dataset['T'])
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    Attention_mask1 = torch.zeros((len(dataset), 1, 34, 34))
    Attention_mask2 = torch.zeros((len(dataset), 1, 16, 16))
    Appearance = torch.zeros((len(dataset), 3, 36, 36))
    print(len(dataset))
    with torch.no_grad():
        val_output = []
        target = []
        for k, (avg, mot, lab) in enumerate(test_loader):
            avg, mot, lab = avg.to(device), mot.to(device), lab.to(device)
            Appearance[k] = avg
            output = DeepPhys(avg, mot)
            mask1, mask2 = output[1], output[2]
            writer.add_image('mask1', mask1[0], k)
            writer.add_image('mask2', mask2[0], k)
            Attention_mask1[k] = mask1
            Attention_mask2[k] = mask2
            print(k)
            val_output.append(output[0].cpu().clone().numpy()[0][0])
            target.append(lab.cpu().clone().numpy()[0][0])
        writer.close()
        io.savemat('./mask/UBFC_train_mask_0524.mat', mdict={'mask1': Attention_mask1.permute(0, 2, 3, 1).numpy(),
                                                             'mask2': Attention_mask2.permute(0, 2, 3, 1).numpy(),
                                                             'Appearance': Appearance.permute(0, 2, 3, 1).numpy(),
                                                             'Pulse_estimate': val_output,
                                                             'target': target})
        print("End Save")
    fs = 30
    low = 0.75 / (0.5 * fs)
    high = 2.5 / (0.5 * fs)
    frame = 300
    [b_pulse, a_pulse] = butter(9, [low, high], btype='band')
    # val_output_normalize = (np.array(val_output) / max(abs(np.array(val_output)))).tolist()
    target = (np.array(target) / max(abs(np.array(target)))).tolist()
    # detrend_pred_100 = detrend(np.cumsum(val_output), 20)
    detrend_pred_200 = detrend(np.cumsum(val_output), 100)
    # detrend_pred_300 = detrend(np.cumsum(val_output_normalize), 300)
    # ------------------------------------pulse-----------------------------------
    # pulse_pred_100 = filtfilt(b_pulse, a_pulse, np.double(detrend_pred_100))
    # pulse_pred_100 = (np.array(pulse_pred_100) / max(abs(np.array(pulse_pred_100)))).tolist()
    # detrend_pred_200 = (np.array(detrend_pred_200) / max(abs(np.array(detrend_pred_200)))).tolist()
    pulse_pred_200 = filtfilt(b_pulse, a_pulse, np.double(detrend_pred_200))
    pulse_pred_200 = (np.array(pulse_pred_200) / max(abs(np.array(pulse_pred_200)))).tolist()
    # pulse_pred_300 = filtfilt(b_pulse, a_pulse, np.double(detrend_pred_300))
    # pulse_pred_300 = (np.array(pulse_pred_300) / max(abs(np.array(pulse_pred_300)))).tolist()
    # pulse_pred_findPeak, _ = find_peaks(pulse_pred[0:frame], distance=10)
    # pulse_target = filtfilt(b_pulse, a_pulse, np.double(target))
    # pulse_target = (np.array(pulse_target) / max(abs(np.array(pulse_target)))).tolist()
    # pulse_target_findPeak, _ = find_peaks(pulse_target[0:frame], distance=10)
    # target_result = len( pulse_target_findPeak[1:])-2
    # inference_result = len(pulse_pred_findPeak)
    # MSE.append(pow(target_result-inference_result, 2))
    # MAE.append(abs(target_result-inference_result))
    print("-----------------------------------------------------------------------------------------------------")
    # print("target : " + str(target_result))
    # print("inference : " + str(inference_result))
    # print("MSE : " + str(sum(MSE)/len(MSE)))
    # print("MAE : " + str(sum(MAE)/len(MAE)))
    # print("rMSE : " + str(sqrt(sum(MSE)/len(MSE))))
    # print(np.corrcoef(pulse_target[100:-100],pulse_pred_100[100:-100])[0, 1])
    print(stats.pearsonr(np.array(target[:]), np.array(pulse_pred_200[:])))
    # print(np.corrcoef(pulse_target[100:-100], pulse_pred_300[100:-100])[0, 1])
    # print(stats.kendalltau(pulse_target,pulse_pred_100).correlation)
    print("-----------------------------------------------------------------------------------------------------")
    plt.rcParams["figure.figsize"] = (14, 5)
    # plt.plot(range(len(pulse_pred_100[0:frame])), pulse_pred_100[0:frame], label='inference_100')
    plt.plot(range(len(pulse_pred_200[0:frame])), pulse_pred_200[0:frame], label='inference')
    # plt.plot(range(len(pulse_pred_300[0:frame])), pulse_pred_300[0:frame], label='inference_300')
    # plt.plot(pulse_pred_findPeak, pulse_pred[pulse_pred_findPeak], 'x')
    plt.plot(range(len(target[0:frame])), target[0:frame], label='target')
    # plt.plot(range(len(target[0:frame])), target[0:frame], label='target')
    # plt.plot(pulse_target_findPeak[1:], pulse_target[pulse_target_findPeak][1:], 'o')
    plt.legend(fontsize='x-large')
    plt.show()
# ------------------------------------resp------------------------------------
# [b_resp, a_resp] = butter(1, [0.08 / fs * 2, 0.5 / fs * 2], btype='bandpass')
# resp_pred = filtfilt(b_resp, a_resp, np.double(detrend_pred))
# resp_target = filtfilt(b_resp, a_resp, np.double(target))
# plt.rcParams["figure.figsize"] = (14, 5)
# plt.plot(range(len(resp_pred[0:300])), resp_pred[0:300], label='inference')
# plt.plot(range(len(target[0:300])), resp_target[0:300], label='target')
# plt.legend(fontsize='x-large')
# plt.show()
