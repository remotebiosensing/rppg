import torch
import os
import numpy as np
import wfdb
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# import matplotlib cmap
import matplotlib.cm as cmap

# heartpy
import heartpy as hp
import heartpy.peakdetection as hp_peak
from heartpy.datautils import rolling_mean
from heartpy.filtering import filter_signal

# scipy
from scipy import signal
# import scipy

# ours
from vid2bp.preprocessing.MIMICdataset import find_available_data, find_idx, read_record, data_aggregator

def signal_respiration_checker(ABP, PPG, threshold=0.9):
    # low pass filter
    ABP = filter_signal(np.squeeze(ABP), cutoff=3, sample_rate=125., order=3, filtertype='lowpass')
    PPG = filter_signal(np.squeeze(PPG), cutoff=3, sample_rate=125., order=3, filtertype='lowpass')
    # normalize
    ABP = 2 * (ABP - np.min(ABP)) / (np.max(ABP) - np.min(ABP)) - 1
    PPG = 2 * (PPG - np.min(PPG)) / (np.max(PPG) - np.min(PPG)) - 1
    # stft
    f_ABP, t_ABP, Zxx_ABP = signal.stft(ABP)
    f_PPG, t_PPG, Zxx_PPG = signal.stft(PPG)
    # cosine similarity
    cosine_similarity = np.sum(np.abs(Zxx_ABP) * np.abs(Zxx_PPG)) / (
            np.sqrt(np.sum(np.abs(Zxx_ABP) ** 2)) * np.sqrt(np.sum(np.abs(Zxx_PPG) ** 2)))
    if cosine_similarity > threshold:
        return True
    else:
        return False


if __name__ == '__main__':
    root_path = '/home/najy/PycharmProjects/rppgs/vid2bp/sample_datasets'
    segment_list = []
    for s in os.listdir(root_path):
        for f in os.listdir(os.path.join(root_path, s)):
            if f.endswith('.hea') and ('_' in f) and ('layout' not in f):
                segment_list.append(os.path.join(root_path, s, f))
    segment_list = [s[:-4] for s in segment_list]

    for segment in segment_list:
        ABP, PPG = read_record(segment, sampfrom=125 * 0, sampto=None)  # input : path without extend, output : ABP, PPG

        if len(ABP) >= 750:
            # raw_ABP = ABP.copy()
            # raw_PPG = PPG.copy()
            ABP = filter_signal(np.squeeze(ABP[:750]), cutoff=3, sample_rate=125., order=2, filtertype='lowpass')
            PPG = filter_signal(np.squeeze(PPG[:750]), cutoff=3, sample_rate=125., order=2, filtertype='lowpass')
            ABP = 2 * (ABP - np.min(ABP)) / (np.max(ABP) - np.min(ABP)) - 1
            PPG = 2 * (PPG - np.min(PPG)) / (np.max(PPG) - np.min(PPG)) - 1
            if np.isnan(np.mean(ABP)) or np.isnan(np.mean(PPG)):
                continue
            else:
                f_ABP, t_ABP, Zxx_ABP = signal.stft(ABP)
                f_PPG, t_PPG, Zxx_PPG = signal.stft(PPG)
                cosine_similarity = np.sum(np.abs(Zxx_ABP) * np.abs(Zxx_PPG)) / (
                            np.sqrt(np.sum(np.abs(Zxx_ABP) ** 2)) * np.sqrt(np.sum(np.abs(Zxx_PPG) ** 2)))
                if cosine_similarity >= 0.95:
                    plt.figure(figsize=(20, 5))
                    # plt.pcolormesh(t_ABP, f_ABP, np.abs(Zxx_ABP), cmap='jet')
                    # plt.pcolormesh
                    plt.plot(ABP)
                    plt.plot(PPG)
                    plt.title(cosine_similarity)
                    plt.show()
