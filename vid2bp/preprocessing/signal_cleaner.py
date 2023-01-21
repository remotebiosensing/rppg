import os
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

import heartpy as hp
import heartpy.peakdetection as hp_peak
from heartpy.datautils import rolling_mean
from heartpy.filtering import filter_signal

from scipy import signal

import wfdb

import vid2bp.preprocessing.utils.signal_utils as su
import vid2bp.preprocessing.utils.math_module as mm
# import vid2bp.preprocessing.mimic3temp as m3t

"""unused"""

def find_channel_idx(path):
    """
    param: path: path of a segment (e.g. /hdd/hdd0/dataset/bpnet/adults/physionet.org/files/mimic3wdb/1.0/30/3001937_11)
    return: idx: index of the ple, abp channel
    """
    record = wfdb.rdrecord(path)
    channel_names = record.sig_name
    ple_idx = [p for p in range(len(channel_names)) if channel_names[p] == 'PLETH'][0]
    abp_idx = [a for a in range(len(channel_names)) if channel_names[a] == 'ABP'][0]

    return ple_idx, abp_idx

def not_flat_signal_detector(target_signal):
    return np.argwhere(np.diff(target_signal) != 0)


def nan_detector(target_signal):
    nan_idx = np.argwhere(np.isnan(target_signal))
    return nan_idx


def discrete_index_detector(target_index):
    return np.where(np.diff(target_index) != 1)


def not_nan_detector(target_signal):
    not_nan_idx = np.argwhere(~np.isnan(target_signal))
    return not_nan_idx


def signals_slice_by_flat(target_signals):
    # input : signal list [signal1, signal2, ...]
    return_signal_list = []
    valid_flag = False
    for target_signal in target_signals:
        not_flat_index = not_flat_signal_detector(target_signal[0]).reshape(-1)
        discrete_index = discrete_index_detector(not_flat_index)
        if len(discrete_index[0]) == 0:
            continue
        else:
            discrete_index = discrete_index[0].reshape(-1)
        if discrete_index[0] != 0:
            discrete_index = np.insert(discrete_index, 0, -1)
        for i in range(len(discrete_index) - 1):
            valid_flag = True
            return_signal_list.append(
                target_signal[:, not_flat_index[discrete_index[i] + 1]:not_flat_index[discrete_index[i + 1]]])
    if valid_flag:
        return True, return_signal_list
    else:
        return False, None


def not_flat_signal_checker(target_signal, t=2, threshold=0.1, slicing=True):
    # return True if not flat
    if slicing:
        target_signal = signal_slicer(target_signal, t=t, overlap=0)
        for sliced_signal in target_signal:
            if np.std(sliced_signal) < threshold:
                return False
    else:
        if np.std(target_signal) < threshold:
            return False
    return True


def not_nan_checker(target_signal):
    return ~np.isnan(target_signal).any()


def nan_interpolator(target_signal):
    # not used
    nan_idx = nan_detector(target_signal)
    nan_idx = nan_idx.reshape(-1)
    for idx in nan_idx:
        target_signal[idx] = np.nanmean(target_signal)
    return target_signal


def signal_slice_by_nan(target_signal):
    not_nan_index = not_nan_detector(target_signal[0]).reshape(-1)
    discrete_index = discrete_index_detector(not_nan_index)
    return_signal_list = []
    if len(discrete_index[0]) == 0:
        return False, None
    else:
        discrete_index = discrete_index[0].reshape(-1)
    if discrete_index[0] != 0:
        discrete_index = np.insert(discrete_index, 0, -1)
    for i in range(len(discrete_index) - 1):
        return_signal_list.append(
            target_signal[:, not_nan_index[discrete_index[i] + 1]:not_nan_index[discrete_index[i + 1]]])
    return True, return_signal_list


def nan_signal_slicer(target_signal):
    nan_idx = nan_detector(target_signal)
    nan_idx = nan_idx.reshape(-1)
    nan_idx = np.append(nan_idx, len(target_signal))
    nan_idx = np.insert(nan_idx, 0, 0)
    nan_idx = nan_idx.reshape(-1, 2)
    for idx in nan_idx:
        yield target_signal[idx[0]:idx[1]]


"""Nan Detecting Modules"""


def nan_checker(target_signal):
    # return if signal has nan
    return np.isnan(target_signal).any(), len(np.where(np.isnan(target_signal))[0])


"""length checking modules"""


def length_checker(target_signal, length):
    if len(target_signal) < length:
        return False
    else:
        return True


"""signal analysing modules"""


def SBP_DBP_arranger(ABP, SBP, DBP):
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


def peak_detector(target_signal, rol_sec, fs=125):
    roll_mean = rolling_mean(target_signal, rol_sec, fs)
    peak_heartpy = hp_peak.detect_peaks(target_signal, roll_mean, ma_perc=20, sample_rate=fs)
    return peak_heartpy['peaklist']


def bottom_detector(target_signal, rol_sec, fs=125):
    target_signal = -target_signal
    roll_mean = rolling_mean(target_signal, rol_sec, fs)
    peak_heartpy = hp_peak.detect_peaks(target_signal, roll_mean, ma_perc=20, sample_rate=fs)
    return peak_heartpy['peaklist']


def correlation_checker(target_signal, reference_signal):
    return np.corrcoef(target_signal, reference_signal)[0, 1]


"""flag signal checking modules"""


def flat_signal_checker(target_signal, t=2, threshold=0.1, slicing=True):
    # return True if flat
    if slicing:
        target_signal = signal_slicer(target_signal, t=t, overlap=0)
        for sliced_signal in target_signal:
            if np.std(sliced_signal) < threshold:
                return True
    else:
        if np.std(target_signal) < threshold:
            return True
    return False


"""t second signal processing modules"""


def signal_slicer(target_signal, fs=125, t=8, overlap=2):
    return_signal_list = []
    while length_checker(target_signal, t * fs):
        return_signal_list.append(target_signal[:t * fs])
        target_signal = target_signal[(t - overlap) * fs:]
    return np.array(return_signal_list)


def signal_shifter(target_signal, gap):
    shifted_signal = np.zeros_like(target_signal)
    if gap > 0:
        shifted_signal[:len(target_signal) - gap] = target_signal[gap:]
        shifted_signal = shifted_signal[:-gap]
    else:
        shifted_signal[-gap:] = target_signal[:len(target_signal) + gap]
        shifted_signal = shifted_signal[-gap:]
    return shifted_signal


def signal_matcher(raw_abp, raw_ple, abp_signal, ple_signal, abp_peaks, ple_peaks, abp_bottoms, ple_bottoms,
                   threshold=0.8):
    best_corr = 0
    best_abp = []
    best_ple = []
    best_gap = 0
    gaps = []
    if len(abp_peaks) > len(ple_peaks):
        for peak_index in range(len(ple_peaks)):
            if peak_index - 1 > 0:
                gaps.append(abp_peaks[peak_index - 1] - ple_peaks[peak_index])

            gaps.append(abp_peaks[peak_index] - ple_peaks[peak_index])

            if peak_index + 1 < len(abp_peaks):
                gaps.append(abp_peaks[peak_index + 1] - ple_peaks[peak_index])
        for gap in gaps:
            matched_ple_signal = signal_shifter(ple_signal, gap)
            if gap > 0:
                matched_abp_signal = abp_signal[:-gap]
            else:
                matched_abp_signal = abp_signal[-gap:]
            if length_checker(matched_abp_signal, 6 * 125) and length_checker(matched_ple_signal, 6 * 125):
                correlation = correlation_checker(matched_abp_signal, matched_ple_signal)
                if correlation > best_corr:
                    best_gap = gap
                    best_corr = correlation
                    best_abp = matched_abp_signal
                    best_ple = matched_ple_signal
        raw_ple = signal_shifter(raw_ple, best_gap)
        if best_gap > 0:
            raw_abp = raw_abp[:-best_gap]
            matched_abp_peaks = abp_peaks[np.argwhere(abp_peaks < len(best_abp)).reshape(-1)]
            matched_ple_peaks = ple_peaks[np.argwhere(ple_peaks > best_gap).reshape(-1)] - best_gap
            matched_abp_bottoms = abp_bottoms[np.argwhere(abp_bottoms < len(best_abp)).reshape(-1)]
            matched_ple_bottoms = ple_bottoms[np.argwhere(ple_bottoms > best_gap).reshape(-1)] - best_gap
        else:
            raw_abp = raw_abp[-best_gap:]
            matched_abp_peaks = abp_peaks[np.argwhere(abp_peaks > -best_gap).reshape(-1)] + best_gap
            matched_ple_peaks = ple_peaks[np.argwhere(ple_peaks < len(best_ple) + best_gap).reshape(-1)]
            matched_abp_bottoms = abp_bottoms[np.argwhere(abp_bottoms > -best_gap).reshape(-1)] + best_gap
            matched_ple_bottoms = ple_bottoms[np.argwhere(ple_bottoms < len(best_ple) + best_gap).reshape(-1)]
    else:
        for peak_index in range(len(abp_peaks)):
            if peak_index - 1 > 0:
                gaps.append(ple_peaks[peak_index - 1] - abp_peaks[peak_index])

            gaps.append(ple_peaks[peak_index] - abp_peaks[peak_index])

            if peak_index + 1 < len(ple_peaks):
                gaps.append(ple_peaks[peak_index + 1] - abp_peaks[peak_index])

        for gap in gaps:
            matched_abp_signal = signal_shifter(abp_signal, gap)
            if gap > 0:
                matched_ple_signal = ple_signal[:-gap]
            else:
                matched_ple_signal = ple_signal[-gap:]
            if length_checker(matched_abp_signal, 6 * 125) and length_checker(matched_ple_signal, 6 * 125):
                correlation = correlation_checker(matched_abp_signal, matched_ple_signal)
                if correlation > best_corr:
                    best_gap = gap
                    best_corr = correlation
                    best_abp = matched_abp_signal
                    best_ple = matched_ple_signal
        raw_abp = signal_shifter(raw_abp, best_gap)
        if best_gap > 0:
            raw_ple = raw_ple[:-best_gap]
            matched_ple_peaks = ple_peaks[np.argwhere(ple_peaks < len(best_ple)).reshape(-1)]
            matched_abp_peaks = abp_peaks[np.argwhere(abp_peaks > best_gap).reshape(-1)] - best_gap
            matched_ple_bottoms = ple_bottoms[np.argwhere(ple_bottoms < len(best_ple)).reshape(-1)]
            matched_abp_bottoms = abp_bottoms[np.argwhere(abp_bottoms > best_gap).reshape(-1)] - best_gap
        else:
            raw_ple = raw_ple[-best_gap:]
            matched_ple_peaks = ple_peaks[np.argwhere(ple_peaks > -best_gap).reshape(-1)] + best_gap
            matched_abp_peaks = abp_peaks[np.argwhere(abp_peaks < len(best_abp) + best_gap).reshape(-1)]
            matched_ple_bottoms = ple_bottoms[np.argwhere(ple_bottoms > -best_gap).reshape(-1)] + best_gap
            matched_abp_bottoms = abp_bottoms[np.argwhere(abp_bottoms < len(best_abp) + best_gap).reshape(-1)]

    if best_corr < threshold:
        return False, None, None, None, None, None, None, None, None, None
    else:
        return True, raw_abp[:750], raw_ple[:750], best_abp[:750], best_ple[:750], matched_abp_peaks[
            np.argwhere(matched_abp_peaks < 750)], \
               matched_ple_peaks[np.argwhere(matched_ple_peaks < 750)], matched_abp_bottoms[
                   np.argwhere(matched_abp_bottoms < 750)], \
               matched_ple_bottoms[np.argwhere(matched_ple_bottoms < 750)], best_corr


if __name__ == "__main__":
    root_path = '/home/najy/PycharmProjects/rppgs/vid2bp/sample_datasets'
    segment_list = []
    for s in os.listdir(root_path):
        for f in os.listdir(os.path.join(root_path, s)):
            if f.endswith('.hea') and ('_' in f) and ('layout' not in f):
                segment_list.append(os.path.join(root_path, s, f))
    segment_list = [s[:-4] for s in segment_list]
    best_abp = []
    best_ple = []
    best_corr = []
    best_abp_peaks = []
    best_ple_peaks = []
    best_abp_bottoms = []
    best_ple_bottoms = []
    total_piece = 0
    interpolated_pieces = 0
    for segment in tqdm(segment_list, desc='segment'):
        ple_idx, abp_idx = find_channel_idx(segment)
        ple, abp = np.squeeze(np.split(wfdb.rdrecord(segment, channels=[ple_idx, abp_idx]).p_signal, 2, axis=1))

        """implement from here"""

        # slice signal by 8 seconds and overlap 2 seconds
        # ensure that the signal length is 8 seconds
        fs = 125
        t = 8
        overlap = 2
        sliced_abp = signal_slicer(abp, fs=fs, t=t, overlap=overlap)
        sliced_ple = signal_slicer(ple, fs=fs, t=t, overlap=overlap)
        total_piece += len(sliced_abp)

        # check if sliced signal is valid
        # if not, remove the signal
        normal_abp = []
        normal_ple = []
        for target_abp, target_ple in zip(sliced_abp, sliced_ple):
            nan_flag_abp, num_nan_abp = nan_checker(target_abp)
            nan_flag_ple, num_nan_ple = nan_checker(target_ple)
            if nan_flag_abp or nan_flag_ple:
                if num_nan_abp < 0.1 * len(target_abp) and num_nan_ple < 0.1 * len(target_ple):
                    target_abp = nan_interpolator(target_abp)
                    target_ple = nan_interpolator(target_ple)
                    interpolated_pieces += 1
                else:
                    continue
            if flat_signal_checker(target_abp) or flat_signal_checker(target_ple):
                continue
            else:
                normal_abp.append(target_abp)
                normal_ple.append(target_ple)

        if len(normal_abp) == 0 or len(normal_ple) == 0:
            continue
        # denoise signal
        lf = 0.5
        hf = 8
        denoised_abp = [filter_signal(target_abp, cutoff=hf, sample_rate=fs, order=2, filtertype='lowpass') for
                        target_abp in normal_abp]
        denoised_ple = [filter_signal(target_ple, cutoff=hf, sample_rate=fs, order=2, filtertype='lowpass') for
                        target_ple in normal_ple]
        # find peak index
        rolling_sec = 1.5
        peak_abp = [peak_detector(target_abp, rolling_sec, fs) for target_abp in denoised_abp]
        peak_ple = [peak_detector(target_ple, rolling_sec, fs) for target_ple in denoised_ple]
        bottom_abp = [bottom_detector(target_abp, rolling_sec, fs) for target_abp in denoised_abp]
        bottom_ple = [bottom_detector(target_ple, rolling_sec, fs) for target_ple in denoised_ple]

        # calculate correlation
        # if correlation is less than threshold, remove the signal
        threshold = 0.8

        for raw_abp, raw_ple, target_abp, target_ple, target_peak_abp, target_peak_ple, target_bottom_abp, target_bottom_ple \
                in zip(normal_abp, normal_ple, denoised_abp, denoised_ple, peak_abp, peak_ple, bottom_abp, bottom_ple):
            # check if peak is valid
            if len(target_peak_abp) == 0 or len(target_peak_ple) == 0:
                continue
            target_peak_abp = np.array(target_peak_abp)
            target_peak_ple = np.array(target_peak_ple)
            target_bottom_abp = np.array(target_bottom_abp)
            target_bottom_ple = np.array(target_bottom_ple)
            # matching signals then get matched peaks, bottoms and correlation
            match_flag, matched_raw_abp, matched_raw_ple, matched_target_abp, matched_target_ple, matched_abp_peaks, \
            matched_ppg_peaks, matched_abp_bottoms, matched_ple_bottoms, matched_corr = signal_matcher(raw_abp, raw_ple,
                                                                                                       target_abp,
                                                                                                       target_ple,
                                                                                                       target_peak_abp,
                                                                                                       target_peak_ple,
                                                                                                       target_bottom_abp,
                                                                                                       target_bottom_ple,
                                                                                                       threshold)
            if match_flag:
                best_abp.append(matched_target_abp)
                best_ple.append(matched_target_ple)
                best_corr.append(matched_corr)
                best_abp_peaks.append(matched_abp_peaks)
                best_ple_peaks.append(matched_ppg_peaks)
                best_abp_bottoms.append(matched_abp_bottoms)
                best_ple_bottoms.append(matched_ple_bottoms)

    print('total matched piece: {}'.format(len(best_abp)))
    print('total piece: {}'.format(total_piece))

    for target_abp, target_ple, target_corr, target_abp_peaks, target_ple_peaks, target_abp_bottoms, target_ple_bottoms in zip(
            best_abp, best_ple, best_corr,
            best_abp_peaks, best_ple_peaks,
            best_abp_bottoms,
            best_ple_bottoms):
        target_abp = (target_abp - np.min(target_abp)) / (np.max(target_abp) - np.min(target_abp))
        target_ple = (target_ple - np.min(target_ple)) / (np.max(target_ple) - np.min(target_ple))
        plt.title('correlation: {}'.format(target_corr))
        plt.plot(target_abp, 'r')
        plt.plot(target_ple, 'b')
        plt.plot(target_abp_peaks, target_abp[target_abp_peaks], 'ro')
        plt.plot(target_ple_peaks, target_ple[target_ple_peaks], 'bo')
        plt.plot(target_abp_bottoms, target_abp[target_abp_bottoms], 'rx')
        plt.plot(target_ple_bottoms, target_ple[target_ple_bottoms], 'bx')
        plt.legend(['ABP', 'PLE', 'ABP peak', 'PLE peak'])
        plt.show()
