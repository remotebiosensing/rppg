import os
import wfdb
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# heartpy
import heartpy as hp
import heartpy.peakdetection as hp_peak
from heartpy.datautils import rolling_mean

# scipy
from scipy import signal

# ours
import vid2bp.preprocessing.utils.signal_utils as su
from vid2bp.preprocessing.MIMICdataset import find_available_data, find_idx, read_record, data_aggregator


def window_wise_heartpy_peak_detection(signal, win_start, win_end, step=0.5, fs=125):
    """
    rolling mean():
        windowsize : [sec], sample_rate : [Hz]

    peak_hartpy():
        ma_perc : the percentage with which to raise the rolling mean, used for fitting detection solutions to data
    """
    peaks = []
    for window in np.arange(win_start, win_end, step=step):
        rol_mean = rolling_mean(signal, window, fs)
        peak_heartpy = hp_peak.detect_peaks(signal, rol_mean, ma_perc=20, sample_rate=fs)
        peaks.append(peak_heartpy)
    return peaks


if __name__ == '__main__':
    root_path = '/home/najy/PycharmProjects/rppgs/vid2bp/sample_dataset'
    segment_list = []
    for s in os.listdir(root_path):
        for f in os.listdir(os.path.join(root_path, s)):
            if f.endswith('.hea') and ('_' in f) and ('layout' not in f):
                segment_list.append(os.path.join(root_path, s, f))
    # segment_list = [root_path + '/' + s for s in os.listdir(root_path) if
    #                 s.endswith('.hea') and ('_' in s) and ('layout' not in s)]
    # strip .hea
    segment_list = [s[:-4] for s in segment_list]
    for segment in segment_list:
        # for i in range()
        signals = read_record(segment, sampfrom=125 * 0, sampto=None)  # input : path without extend, output : ABP, PPG
        if len(signals) >= 750:
            signals = signals[:750]
            ABP, PPG = su.sig_spliter(signals)
            ABP = np.squeeze(ABP)
            PPG = np.squeeze(PPG)
            if np.isnan(np.mean(ABP)) or np.isnan(PPG.any()):
                continue
            else:
                # PPG = np.squeeze(PPG)

                ### rolling mean by 0.75 sec ###
                rol_mean = rolling_mean(ABP, 0.75, 125)
                peak_heartpy = hp_peak.detect_peaks(ABP, rol_mean, ma_perc=20, sample_rate=125)
                plt.plot(ABP, label='ABP')
                plt.plot(peak_heartpy, ABP, 'x', label='heartpy', color='red')
                plt.legend()
                plt.show()
                pass
        else:
            continue
    """scipy"""
    # peak_scipy, property_scipy = signal.find_peaks(ABP, height=np.max(ABP) - np.std(ABP))

    """heartpy"""
    ### window-wise rolling mean ###
    # heartpy_peakdict = window_wise_heartpy_peak_detection(ABP, win_start=0.5, win_end=2.5, step=0.25, fs=125)
    # heartpy_peaks = []
    # for i in range(len(heartpy_peakdict)):
    #     heartpy_peaks.append(heartpy_peakdict[i]['peaklist'])
    ############################



    ### plot signal with peaks ###
    # plt.figure(figsize=(10, 5))
    # plt.plot(ABP, label='ABP')
    # plt.plot(PPG, label='PPG')
    # plt.plot(peak_scipy, ABP[peak_scipy], 'x', label='scipy')
    # color_list = list(mcolors.TABLEAU_COLORS.keys())
    # for i in range(len(heartpy_peaks)):
    #     plt.plot(ABP, label='ABP')
    #     plt.plot(heartpy_peaks[i], ABP[heartpy_peaks[i]], 'x', label='heartpy', color=color_list[i])
    #     plt.legend()
    #     plt.show()
    ############################