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
import vid2bp.preprocessing.mimic3temp as m3t

"""Nan Detecting Modules"""


def nan_detector(target_signal):
    nan_idx = np.argwhere(np.isnan(target_signal))
    return nan_idx


def not_nan_checker(target_signal):
    return ~np.isnan(target_signal).any()


def nan_checker(target_signal):
    # return if signal has nan
    return np.isnan(target_signal).any()


def not_nan_detector(target_signal):
    not_nan_idx = np.argwhere(~np.isnan(target_signal))
    return not_nan_idx


def discrete_index_detector(target_index):
    return np.where(np.diff(target_index) != 1)


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


def nan_interpolator(target_signal):
    # not used
    nan_idx = nan_detector(target_signal)
    nan_idx = nan_idx.reshape(-1)
    for idx in nan_idx:
        target_signal[idx] = np.nanmean(target_signal)
    return target_signal


"""length checking modules"""


def length_checker(target_signal, length):
    if len(target_signal) < length:
        return False
    else:
        return True


"""signal analysing modules"""


def denoiser(target_signal, fs=125, order=5, cutoff=0.5):
    b, a = signal.butter(order, cutoff, 'low', fs=fs)
    filtered_signal = signal.filtfilt(b, a, target_signal)
    return filtered_signal


def peak_detector(target_signal, fs=125, threshold=0.5):
    peak_idx = hp_peak.detect_peaks(target_signal, fs, threshold)
    peak_idx = np.array(peak_idx)
    return peak_idx


def signal_shifter(target_signal, shift):
    shifted_signal = np.roll(target_signal, shift)
    return shifted_signal


def correlation_checker(target_signal, reference_signal):
    corr = np.correlate(target_signal, reference_signal, mode='full')
    return corr


"""flag signal checking modules"""


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


def not_flat_signal_detector(target_signal):
    return np.argwhere(np.diff(target_signal) != 0)


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


"""t second signal processing modules"""


def signal_length_slicer(target_signal, length):
    while length_checker(target_signal, length):
        yield target_signal[:length]
        target_signal = target_signal[length:]


def signal_slicer(target_signal, fs=125, t=8, overlap=2):
    return_signal_list = []
    while length_checker(target_signal, t * fs):
        return_signal_list.append(target_signal[:t * fs])
        target_signal = target_signal[(t - overlap) * fs:]
    return return_signal_list


if __name__ == "__main__":
    root_path = '/home/najy/PycharmProjects/rppgs/vid2bp/sample_datasets'
    segment_list = []
    for s in os.listdir(root_path):
        for f in os.listdir(os.path.join(root_path, s)):
            if f.endswith('.hea') and ('_' in f) and ('layout' not in f):
                segment_list.append(os.path.join(root_path, s, f))
    segment_list = [s[:-4] for s in segment_list]
    for segment in tqdm(segment_list):
        ple_idx, abp_idx = m3t.find_channel_idx(segment)
        ple, abp = np.squeeze(np.split(wfdb.rdrecord(segment, channels=[ple_idx, abp_idx]).p_signal, 2, axis=1))

        """implement from here"""

        # slice signal by 8 seconds and overlap 2 seconds
        # ensure that the signal length is 8 seconds
        sliced_abp = signal_slicer(abp)
        sliced_ple = signal_slicer(ple)

        # check if sliced signal is valid
        # if not, remove the signal
        for target_abp, target_ple in zip(sliced_abp, sliced_ple):
            if nan_checker(target_abp) or nan_checker(target_ple)\
                    or flat_signal_checker(target_abp) or flat_signal_checker(target_ple):
                sliced_abp.remove(target_abp)
                sliced_ple.remove(target_ple)

        # denoise signal
        denoised_abp = [denoiser(target_abp) for target_abp in sliced_abp]
        denoised_ple = [denoiser(target_ple) for target_ple in sliced_ple]

        # add index to ple, abp
        ple = np.vstack([ple, np.arange(len(ple))])
        abp = np.vstack([abp, np.arange(len(abp))])
        # slice signal by nan
        nan_flag_abp, abp_list = signal_slice_by_nan(abp)
        nan_flag_ple, ple_list = signal_slice_by_nan(ple)
        if nan_flag_abp and nan_flag_ple:
            # check flat signal
            flat_flag_abp, abp_list = signals_slice_by_flat(abp_list)
            flat_flag_ple, ple_list = signals_slice_by_flat(ple_list)
            if flat_flag_abp and flat_flag_ple:
                # check signal length
                # abp_list = [s for s in abp_list if length_checker(s, 6 * 125)]
                # ple_list = [s for s in ple_list if length_checker(s, 6 * 125)]
                for abp_idx in range(0, len(abp), 750):
                    plt.plot(abp[0, abp_idx:abp_idx + 750])
                    plt.show()
                print(len(abp_list), len(ple_list))
