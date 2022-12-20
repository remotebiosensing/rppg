import os
import wfdb
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# heartpy
import heartpy as hp
import heartpy.peakdetection as hp_peak
from heartpy.datautils import rolling_mean
from heartpy.filtering import filter_signal

# scipy
from scipy import signal

# ours
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


def SBP_detection(signal, rolling_sec=0.75, fs=125):
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


def match_signal(ABP, PPG, SBP, DBP, PPG_peak, PPG_low):
    if PPG_peak[0] < SBP[0]:
        matched_ABP = ABP[SBP[0]:]
        matched_PPG, gap_size = PPG[PPG_peak[0]:len(matched_ABP) + PPG_peak[0]], PPG_peak[0] - SBP[0]
    else:
        matched_PPG = PPG[PPG_peak[0]:]
        matched_ABP, gap_size = ABP[SBP[0]:len(matched_PPG) + SBP[0]], PPG_peak[0] - SBP[0]

    if gap_size >= 0:
        gap_size = SBP[0]
        SBP = [SBP[x] - gap_size for x in range(len(SBP)) if
               0 <= SBP[x] - gap_size < len(matched_ABP)]
        DBP = [DBP[x] - gap_size for x in range(len(DBP)) if
               0 <= DBP[x] - gap_size < len(matched_ABP)]
        gap_size = PPG_peak[0]
        PPG_peak = [PPG_peak[x] - gap_size for x in range(len(PPG_peak)) if
                    0 <= PPG_peak[x] - gap_size < len(matched_PPG)]
        PPG_low = [PPG_low[x] - gap_size for x in range(len(PPG_low)) if
                   0 <= PPG_low[x] - gap_size < len(matched_PPG)]
    else:
        gap_size = PPG_peak[0]

        PPG_peak = [PPG_peak[x] - gap_size for x in range(len(PPG_peak)) if
                    len(matched_PPG) > PPG_peak[x] - gap_size >= 0]
        PPG_low = [PPG_low[x] - gap_size for x in range(len(PPG_low)) if
                   len(matched_PPG) > PPG_low[x] - gap_size >= 0]
        gap_size = SBP[0]
        SBP = [SBP[x] - gap_size for x in range(len(SBP)) if
               len(matched_PPG) > SBP[x] - gap_size >= 0]
        DBP = [DBP[x] - gap_size for x in range(len(DBP)) if
               len(matched_PPG) > DBP[x] - gap_size >= 0]

    return matched_ABP, matched_PPG, gap_size, SBP, DBP, PPG_peak, PPG_low


def signals_rolling_mean(ABP, PPG, rolling_sec, fs=125):
    # rolling mean for find proper trend
    ABP_rolling_mean = rolling_mean(ABP, rolling_sec, fs)
    PPG_rolling_mean = rolling_mean(PPG, rolling_sec, fs)
    return ABP_rolling_mean, PPG_rolling_mean


def plot_signal_with_props(ABP, PPG, SBP, DBP, PPG_peak, PPG_low, ABP_rolling_mean, PPG_rolling_mean,
                           title='signal with properties'):
    plt.figure(figsize=(20, 5))
    plt.plot(ABP)
    plt.plot(PPG)
    plt.plot(SBP, ABP[SBP], 'ro')
    plt.plot(DBP, ABP[DBP], 'bo')
    plt.plot(PPG_peak, PPG[PPG_peak], 'go')
    plt.plot(PPG_low, PPG[PPG_low], 'yo')
    plt.plot(ABP_rolling_mean, 'g', linestyle='--')
    plt.plot(PPG_rolling_mean, 'y', linestyle='--')
    plt.title(title)
    plt.legend(['ABP', 'PPG', 'SBP', 'DBP', 'PPG_peak', 'PPG_low', 'ABP_rolling_mean', 'PPG_rolling_mean'])
    plt.show()


def SBP_DBP_filter(ABP, SBP, DBP):
    i = 0
    total = len(SBP) - 1
    while i < total:
        flag = False
        # Distinguish SBP[i] < DBP < SBP[i+1]
        for idx_dbp in DBP:
            # Normal situation
            if (SBP[i] < idx_dbp) and (idx_dbp < SBP[i + 1]):
                flag = True
                break
            # abnormal situation
        if flag:
            i += 1
        else:
            # compare peak value
            # delete smaller one @SBP
            if ABP[SBP[i]] < ABP[SBP[i + 1]]:
                SBP = np.delete(SBP, i)
            else:
                SBP = np.delete(SBP, i + 1)
            total -= 1

    i = 0
    total = len(DBP) - 1
    while i < total:
        flag = False
        # Distinguish DBP[i] < SBP < DBP[i+1]
        for idx_sbp in SBP:
            # Normal situation
            if (DBP[i] < idx_sbp) and (idx_sbp < DBP[i + 1]):
                flag = True
                break
        # normal situation
        if flag:
            i += 1
        # abnormal situation, there is no SBP between DBP[i] and DBP[i+1]
        else:
            # compare peak value
            # delete bigger one @DBP
            if ABP[DBP[i]] < ABP[DBP[i + 1]]:
                DBP = np.delete(DBP, i + 1)
            else:
                DBP = np.delete(DBP, i)
            total -= 1

    return SBP, DBP


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

            ABP = filter_signal(np.squeeze(ABP[:750]), cutoff=3, sample_rate=125., order=2, filtertype='lowpass')
            PPG = filter_signal(np.squeeze(PPG[:750]), cutoff=3, sample_rate=125., order=2, filtertype='lowpass')

            # Normalization
            ABP = 2 * (ABP - np.min(ABP)) / (np.max(ABP) - np.min(ABP)) - 1
            PPG = 2 * (PPG - np.min(PPG)) / (np.max(PPG) - np.min(PPG)) - 1

            if np.isnan(np.mean(ABP)) or np.isnan(np.mean(PPG)):
                continue
            else:
                ### rolling mean by 'rolling_sec' sec ###
                rolling_sec = 0.75
                r_rolling_sec = 0.75
                SBP = SBP_detection(ABP, rolling_sec)
                DBP = DBP_detection(ABP, rolling_sec)
                SBP, DBP = SBP_DBP_filter(ABP, SBP, DBP)
                PPG_peak = SBP_detection(PPG, rolling_sec)
                PPG_low = DBP_detection(PPG, rolling_sec)
                PPG_peak, PPG_low = SBP_DBP_filter(PPG, PPG_peak, PPG_low)

                if len(SBP) > 2:
                    matched_ABP, matched_PPG, gap_size, SBP, DBP, PPG_peak, PPG_low = match_signal(ABP, PPG, SBP, DBP,
                                                                                                   PPG_peak, PPG_low)
                else:
                    continue

                if len(SBP) > 2:
                    r_rolling_sec = 0.5

                ABP_rolling_mean, PPG_rolling_mean = signals_rolling_mean(matched_ABP, matched_PPG, r_rolling_sec)

                ABP_rolling_mean = 2 * (ABP_rolling_mean - np.min(ABP_rolling_mean)) / (
                        np.max(ABP_rolling_mean) - np.min(ABP_rolling_mean)) - 1
                PPG_rolling_mean = 2 * (PPG_rolling_mean - np.min(PPG_rolling_mean)) / (
                        np.max(PPG_rolling_mean) - np.min(PPG_rolling_mean)) - 1

                # correlation = r(ABP_rolling_mean, PPG_rolling_mean)
                # correlation = (np.square(ABP_rolling_mean - PPG_rolling_mean)).mean(axis=0)
                correlation = np.mean(np.corrcoef(ABP_rolling_mean, PPG_rolling_mean))

                plot_signal_with_props(matched_ABP, matched_PPG, SBP, DBP, PPG_peak, PPG_low, ABP_rolling_mean,
                                       PPG_rolling_mean,
                                       title='corrcoef : {:.2f}'.format(correlation)
                                             + ' SBP : {:.2f}'.format(np.mean(raw_ABP[SBP]))
                                             + ' DBP : {:.2f}'.format(np.mean(raw_ABP[DBP]))
                                             + ' rolling_sec : {}'.format(rolling_sec)
                                             + ' r_rolling_sec : {}'.format(r_rolling_sec)
                                             + ' gap_size : {}'.format(gap_size))
        else:
            continue
