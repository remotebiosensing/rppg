import numpy as np
from scipy import io
import inference_preprocess
from scipy.signal import butter, filtfilt, find_peaks
from scipy.sparse import eye, spdiags
import matplotlib.pyplot as plt
import h5py
import cv2
import torch

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def translate_spdiags(T1):
    # Matlab : D2=spdiags(ones(T-2,1)*[1 -2 1],[0:2],T-2,T)
    D1 = spdiags((np.ones((T1 - 2, 1)) * [1, -2, 1]).T, [0, 1, 2], T1 - 2, T1)
    a = np.zeros((T1 - 2, T1))
    a[T1 - 4, T1 - 2] = 1
    a[T1 - 3, T1 - 2] = -2
    a[T1 - 3, T1 - 1] = 1
    D1 = D1 + a
    return D1


# Data Load - mask.mat file
mat_file = io.loadmat('./mask/UBFC_train_mask_0524.mat')
Attention_mask1 = mat_file['mask1']
# Attention_mask1 = Attention_mask1[:10000]
Attention_mask2 = mat_file['mask2']
# Attention_mask2 = Attention_mask2[:10000]
Appearance = mat_file['Appearance']
# Appearance = Appearance[:10000]
Pulse_estimate = mat_file['Pulse_estimate']
Pulse_estimate = Pulse_estimate.T
# Pulse_estimate = Pulse_estimate[:10000]
print('Data Load')
print(len(Attention_mask1))
# Inverse Attention Mask Parameter
mask_noise = np.zeros_like(Attention_mask1)
L = 34
fs = 30
[b_pulse, a_pulse] = butter(9, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
print('butter worth Filter')

# Normalize
for i in range(len(Attention_mask1)):
    mask_tmp = Attention_mask1[i] - np.min(Attention_mask1[i])
    mask_tmp = mask_tmp / np.max(Attention_mask1[i])

    mask_noise_tmp = mask_tmp
    threshold = 0.1

    mask_noise_tmp = np.where(mask_noise_tmp > threshold, 0, 1.0)

    mask_noise[i] = mask_noise_tmp

mask_noise = np.squeeze(mask_noise)
print('End Normalize')

for Channel in range(0, 3):
    mask_noise_i_tmp = mask_noise
    mask_noise_channel = np.zeros((len(Pulse_estimate), 34, 34))

    for frame in range(len(Appearance)):
        dXsub_resize = cv2.resize(Appearance[frame], dsize=(34, 34), interpolation=cv2.INTER_CUBIC)
        dXsub_resize = np.where(dXsub_resize > 1, 1, dXsub_resize)
        dXsub_resize = np.where(dXsub_resize < 1 / 255, 1 / 255, dXsub_resize)

        # Multiply
        mask_noise_channel[frame, :, :] = np.squeeze(mask_noise_i_tmp[frame]) * dXsub_resize[:, :, Channel]

    mask_noise_channel_mean = np.mean(np.mean(mask_noise_channel, axis=2), axis=1)
    yptest_sub1 = np.cumsum(Pulse_estimate)
    print('End Multiply')

    pulse_lambda = 100
    T1 = len(yptest_sub1)
    I1 = eye(T1).toarray()
    I1 = np.asmatrix(I1)
    D1 = np.asmatrix(translate_spdiags(T1))
    print('D1')
    sr1 = np.transpose(np.array([yptest_sub1]))
    D1_MA = np.linalg.lstsq(np.transpose((I1 + (pulse_lambda ** 2) * D1.T * D1)), sr1)[0]
    print('D1_MA')
    # D1_MA = inference_preprocess.detrend(yptest_sub1, 100)
    # print('D1_MA')
    temp1 = I1 * sr1 - D1_MA
    print('temp1')
    nZ1 = (temp1 - np.mean(temp1)) / np.std(temp1, ddof=1)
    nZ1 = np.squeeze(np.array(nZ1, dtype=np.float))
    print('nZ1')
    yptest_sub2 = filtfilt(b_pulse, a_pulse, np.double(nZ1))
    print('filtfilt')

    # plt.rcParams["figure.figsize"] = (14, 5)
    # plt.plot(range(len(nZ1[0:300])), nZ1[0:300], label='Pulse Estimate')
    # plt.plot(range(len(yptest_sub2[0:300])), yptest_sub2[0:300], label='filter')
    # plt.legend(fontsize='x-large')
    # plt.show()

    T2 = len(mask_noise_channel_mean)
    I2 = eye(T2).toarray()
    I2 = np.asmatrix(I2)
    D2 = np.asmatrix(translate_spdiags(T2))
    sr2 = np.transpose(np.array([mask_noise_channel_mean]))
    D2_MA = np.linalg.lstsq(np.transpose((I2 + (pulse_lambda ** 2) * D2.T * D2)), sr2)[0]
    # D2_MA = inference_preprocess.detrend(mask_noise_channel, 100)
    temp2 = I2 * sr2 - D2_MA
    nZ2 = (temp2 - np.mean(temp2)) / np.std(temp2, ddof=1)
    nZ2 = np.squeeze(np.array(nZ2, dtype=np.float))
    mask_noise_channel_mean2 = filtfilt(b_pulse, a_pulse, np.double(nZ2))

    # plt.rcParams["figure.figsize"] = (14, 5)
    # plt.plot(range(len(nZ2[0:300])), nZ2[0:300], label='Noise Estimate')
    # plt.plot(range(len(mask_noise_channel_mean2[0:300])),
    #          mask_noise_channel_mean2[0:300], label='filter')
    # plt.legend(fontsize='x-large')
    # plt.show()

    # rename variables to red, green, channel:
    if Channel == 0:
        mask_noise_red = mask_noise_channel
        mask_noise_red_mean = mask_noise_channel_mean
        mask_noise_red_mean2 = mask_noise_channel_mean2
        io.savemat('./Noise_estimate/Noise_estimate_red_total.mat', mdict={'mask_noise_red_mean': mask_noise_red_mean2})
        print("red!")

    elif Channel == 1:
        mask_noise_green = mask_noise_channel
        mask_noise_green_mean = mask_noise_channel_mean
        mask_noise_green_mean2 = mask_noise_channel_mean2
        io.savemat('./Noise_estimate/Noise_estimate_green_total.mat', mdict={'mask_noise_green_mean': mask_noise_green_mean2})
        print("green!")

    elif Channel == 2:
        mask_noise_blue = mask_noise_channel
        mask_noise_blue_mean = mask_noise_channel_mean
        mask_noise_blue_mean2 = mask_noise_channel_mean2
        io.savemat('./Noise_estimate/Noise_estimate_blue_total.mat', mdict={'mask_noise_blue_mean': mask_noise_blue_mean2})
        io.savemat('./Noise_estimate/Noise_estimate_Pulse_estimate_total.mat', mdict={'Noise_estimate_Pulse_estimate': yptest_sub2})

        print('blue!')

# n_examples = 1
#
# noise_red = np.transpose(mask_noise_red_mean2)
# noise_green = np.transpose(mask_noise_green_mean2)
# noise_blue = np.transpose(mask_noise_blue_mean2)
# noise_signal = np.transpose(yptest_sub2)
#
# noise_red = noise_red / max(abs(noise_red))
# noise_green = noise_green / max(abs(noise_green))
# noise_blue = noise_blue / max(abs(noise_blue))
# noise_signal = noise_signal / max(abs(noise_signal))
#
# n_numbers = 4  # number of features - 1 signal + 3 noise estimates will be concatenated together
# n_chars = 1
# n_batch = 32  # batch size
# n_epoch = 11  # number of epochs
# n_examples = 60  # length of the signals input to the LSTM
#
# signal_length = len(noise_red)
# n_samples = len(range(0, signal_length - (n_examples - 1), int(n_examples / 2)))
#
# Xtrain1 = np.zeros((n_samples, n_examples), dtype=np.float32)
# Xtrain2 = np.zeros((n_samples, n_examples), dtype=np.float32)
# Xtrain3 = np.zeros((n_samples, n_examples), dtype=np.float32)
# Xtrain4 = np.zeros((n_samples, n_examples), dtype=np.float32)
# Xtrain = np.zeros((n_samples, n_examples, 4), dtype=np.float32)
# Ytrain = np.zeros((n_samples, n_examples), dtype=np.float32)
#
# Xtest1 = np.zeros((n_samples, n_examples), dtype=np.float32)
# Xtest2 = np.zeros((n_samples, n_examples), dtype=np.float32)
# Xtest3 = np.zeros((n_samples, n_examples), dtype=np.float32)
# Xtest4 = np.zeros((n_samples, n_examples), dtype=np.float32)
# Xtest = np.zeros((n_samples, n_examples, 4), dtype=np.float32)
# Ytest = np.zeros((n_samples, n_examples), dtype=np.float32)
#
# count = 0
# for j in range(0, len(noise_signal) - (n_examples - 1), int(n_examples / 2)):
#     Xtrain1[count, 0:n_examples] = np.transpose(noise_red[j:j + n_examples])
#     Xtrain2[count, 0:n_examples] = np.transpose(noise_green[j:j + n_examples])
#     Xtrain3[count, 0:n_examples] = np.transpose(noise_blue[j:j + n_examples])
#     Xtrain4[count, 0:n_examples] = np.transpose(noise_signal[j:j + n_examples])
#     count = count + 1
#
# Xtrain[:, :, 0] = Xtrain1
# Xtrain[:, :, 1] = Xtrain2
# Xtrain[:, :, 2] = Xtrain3
# Xtrain[:, :, 3] = Xtrain4
#
# Ytrain = Ytrain.reshape(Ytrain.shape[0], Ytrain.shape[1], 1)
# Ytrain = Ytrain.reshape(Ytrain.shape[0], Ytrain.shape[1], 1)
# print('test')
