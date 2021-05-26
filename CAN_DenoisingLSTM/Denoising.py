import numpy as np
from scipy import io
from torch.utils.data import DataLoader
import inference_preprocess
from scipy.signal import butter, filtfilt, find_peaks
from scipy.sparse import eye, spdiags
import matplotlib.pyplot as plt
import h5py
import cv2
import torch

from LSTM_DATA import timeseries

mat_file = io.loadmat('./Noise_estimate/Noise_estimate_red.mat')
mask_noise_red = mat_file['mask_noise_red_mean']
mat_file = io.loadmat('./Noise_estimate/Noise_estimate_blue.mat')
mask_noise_blue = mat_file['mask_noise_blue_mean']
mat_file = io.loadmat('./Noise_estimate/Noise_estimate_green.mat')
mask_noise_green = mat_file['mask_noise_green_mean']
mat_file = io.loadmat('./Noise_estimate/Noise_estimate_Pulse_estimate.mat')
mask_noise_Pulse_estimate = mat_file['Noise_estimate_Pulse_estimate']
mat_file = io.loadmat('./mask/UBFC_train_mask_0524.mat')
GT = mat_file['target']
GT = GT.T[:10000]
n_examples = 1

noise_red = np.transpose(mask_noise_red)
noise_green = np.transpose(mask_noise_green)
noise_blue = np.transpose(mask_noise_blue)
noise_signal = np.transpose(mask_noise_Pulse_estimate)

noise_red = noise_red / max(abs(noise_red))
noise_green = noise_green / max(abs(noise_green))
noise_blue = noise_blue / max(abs(noise_blue))
noise_signal = noise_signal / max(abs(noise_signal))
GT = GT / max(abs(GT))

n_numbers = 4  # number of features - 1 signal + 3 noise estimates will be concatenated together
n_chars = 1
n_batch = 32  # batch size
n_epoch = 11  # number of epochs
n_examples = 60  # length of the signals input to the LSTM

signal_length = len(noise_red)
n_samples = len(range(0, signal_length - (n_examples - 1), int(n_examples / 2)))

Xtrain1 = np.zeros((n_samples, n_examples), dtype=np.float32)
Xtrain2 = np.zeros((n_samples, n_examples), dtype=np.float32)
Xtrain3 = np.zeros((n_samples, n_examples), dtype=np.float32)
Xtrain4 = np.zeros((n_samples, n_examples), dtype=np.float32)
Xtrain = np.zeros((n_samples, n_examples, 4), dtype=np.float32)
Ytrain = np.zeros((n_samples, n_examples), dtype=np.float32)

Xtest1 = np.zeros((n_samples, n_examples), dtype=np.float32)
Xtest2 = np.zeros((n_samples, n_examples), dtype=np.float32)
Xtest3 = np.zeros((n_samples, n_examples), dtype=np.float32)
Xtest4 = np.zeros((n_samples, n_examples), dtype=np.float32)
Xtest = np.zeros((n_samples, n_examples, 4), dtype=np.float32)
Ytest = np.zeros((n_samples, n_examples), dtype=np.float32)

count = 0
for j in range(0, len(noise_signal) - (n_examples - 1), int(n_examples / 2)):
    Xtrain1[count, 0:n_examples] = np.transpose(noise_red[j:j + n_examples])
    Xtrain2[count, 0:n_examples] = np.transpose(noise_green[j:j + n_examples])
    Xtrain3[count, 0:n_examples] = np.transpose(noise_blue[j:j + n_examples])
    Xtrain4[count, 0:n_examples] = np.transpose(noise_signal[j:j + n_examples])
    Xtrain4[count, 0:n_examples] = np.transpose(noise_signal[j:j + n_examples])
    Ytrain[count, 0:n_examples] = np.transpose(GT[j:j + n_examples])
    count = count + 1

Xtrain[:, :, 0] = Xtrain1
Xtrain[:, :, 1] = Xtrain2
Xtrain[:, :, 2] = Xtrain3
Xtrain[:, :, 3] = Xtrain4

Ytrain = Ytrain.reshape(Ytrain.shape[0], Ytrain.shape[1], 1)

io.savemat('./preprocessing/LSTM_dataset/UBFC_10000_train.mat', mdict={'train': Xtrain,
                                                                       'test' : Ytrain})

print('test')
