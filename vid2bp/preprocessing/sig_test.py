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
# from scipy import signal

# ours
from vid2bp.preprocessing.MIMICdataset import find_available_data, find_idx, read_record, data_aggregator
# from vid2bp.nets.loss.loss import r


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


def SBP_detection(signal, rolling_sec, fs=125):
    roll_mean = rolling_mean(signal, rolling_sec, fs)
    peak_heartpy = hp_peak.detect_peaks(signal, roll_mean, ma_perc=20, sample_rate=fs)
    return peak_heartpy['peaklist']


def DBP_detection(signal, rolling_sec, fs=125):
    signal = -signal
    roll_mean = rolling_mean(signal, rolling_sec, fs)
    peak_heartpy = hp_peak.detect_peaks(signal, roll_mean, ma_perc=20, sample_rate=fs)
    return peak_heartpy['peaklist']


def PPG_peak_detection(PPG, rolling_sec, fs=125):
    PPG_rolling_mean = rolling_mean(PPG, rolling_sec, fs)
    peak_heartpy = hp_peak.detect_peaks(PPG, PPG_rolling_mean, ma_perc=20, sample_rate=fs)
    return peak_heartpy['peaklist']


def match_signal(ABP, PPG, SBP, PPG_peak):
    """
    1. pivot -> bad synchronization
        BP_pviot = SBP[np.argmax(ABP[SBP])]
        PPG_pviot = PPG_peak[np.argmax(PPG[PPG_peak])]
        gap = PPG_pviot - BP_pviot
    2. second, third peak -> bad synchronization
        gap = PPG_peak[2] - SBP[2]
    3. minimum gap of around second peak -> bad synchronization
        diff = PPG[PPG_peak[-1]] - ABP[SBP[-1]]
        gap = PPG_peak[-1] - SBP[-1]
        for i in range(3):
            tmp_diff = PPG[PPG_peak[i]] - ABP[SBP[2]]
            if abs(tmp_diff) < abs(diff):
                diff = tmp_diff
                gap = PPG_peak[i] - SBP[2]
    """
    # previous version
    # matched_PPG = np.zeros_like(ABP)
    # gap = abs(PPG_peak[0] - SBP[0])
    # matched_PPG[:len(PPG) - gap] = PPG[gap:]

    matched_PPG = np.zeros_like(ABP)
    gap = abs(PPG_peak[0] - SBP[0])

    matched_PPG[:len(matched_PPG)-gap] = PPG[gap:]
    # matched_PPG = matched_PPG[:len(matched_PPG)-gap]
    matched_PPG = matched_PPG[:-gap]
    matched_ABP = ABP[:-gap]

    return matched_ABP, matched_PPG


def signals_rolling_mean(ABP, PPG, rolling_sec, fs=125):
    # rolling mean for find proper trend
    ABP_rolling_mean = rolling_mean(ABP, rolling_sec, fs)
    PPG_rolling_mean = rolling_mean(PPG, rolling_sec, fs)
    return ABP_rolling_mean, PPG_rolling_mean


def plot_signal_with_props(ABP, PPG, SBP, DBP, ABP_rolling_mean, PPG_rolling_mean, title='signal with properties'):
    plt.figure(figsize=(20, 5))
    plt.plot(ABP)
    plt.plot(PPG)
    plt.plot(SBP, ABP[SBP], 'ro')
    plt.plot(DBP, ABP[DBP], 'bo')
    plt.plot(ABP_rolling_mean, 'g')
    plt.plot(PPG_rolling_mean, 'y')
    plt.title(title)
    plt.legend(['ABP', 'PPG', 'SBP', 'DBP', 'ABP_rolling_mean', 'PPG_rolling_mean'])
    plt.show()


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
            raw_ABP = ABP.copy()
            raw_PPG = PPG.copy()
            # Normalization
            ABP = np.squeeze(ABP[:750])
            PPG = np.squeeze(PPG[:750])

            ABP = (ABP - np.min(ABP)) / (np.max(ABP) - np.min(ABP))
            PPG = (PPG - np.min(PPG)) / (np.max(PPG) - np.min(PPG))
            # ABP = (ABP - np.mean(ABP)) / np.std(ABP)
            # PPG = (PPG - np.mean(PPG)) / np.std(PPG)

            if np.isnan(np.mean(ABP)) or np.isnan(np.mean(PPG)):
                continue
            else:
                ### rolling mean by 'rolling_sec' sec ###
                rolling_sec = 1.5
                r_rolling_sec = 3
                SBP = SBP_detection(ABP, rolling_sec)
                DBP = DBP_detection(ABP, rolling_sec)

                PPG_peak = PPG_peak_detection(PPG, rolling_sec)
                matched_ABP, matched_PPG = match_signal(ABP, PPG, SBP, PPG_peak)
                if len(SBP) > 2:
                    r_rolling_sec = (SBP[2] - SBP[1]) * 3 / 125
                ABP_rolling_mean, PPG_rolling_mean = signals_rolling_mean(matched_ABP, matched_PPG, r_rolling_sec)
                # ABP_rolling_mean = ABP_rolling_mean[:SBP[-1]]
                # PPG_rolling_mean = PPG_rolling_mean[:SBP[-1]]

                ABP_rolling_mean = (ABP_rolling_mean - np.min(ABP_rolling_mean)) / \
                                   (np.max(ABP_rolling_mean) - np.min(ABP_rolling_mean))
                PPG_rolling_mean = (PPG_rolling_mean - np.min(PPG_rolling_mean)) / \
                                   (np.max(PPG_rolling_mean) - np.min(PPG_rolling_mean))

                # correlation = r(ABP_rolling_mean, PPG_rolling_mean)
                correlation = (np.square(ABP_rolling_mean - PPG_rolling_mean)).mean(axis=0)
                # correlation = np.mean(np.corrcoef(ABP_rolling_mean, PPG_rolling_mean))

                ### plot ###
                plot_signal_with_props(matched_ABP, matched_PPG, SBP, DBP, ABP_rolling_mean, PPG_rolling_mean,
                                       title='correlation : {:.2f}'.format(correlation)
                                             + ' SBP : {:.2f}'.format(np.mean(raw_ABP[SBP]))
                                             + ' DBP : {:.2f}'.format(np.mean(raw_ABP[DBP]))
                                             + ' rolling_sec : {}'.format(rolling_sec)
                                             + ' r_rolling_sec : {}'.format(r_rolling_sec))
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
