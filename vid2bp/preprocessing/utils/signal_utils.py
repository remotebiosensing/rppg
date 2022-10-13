import numpy as np
from scipy import signal
from sklearn import preprocessing
import matplotlib.pyplot as plt
from vid2bp.nets.loss.loss import r
from tqdm import tqdm

import json

with open("/home/paperc/PycharmProjects/Pytorch_rppgs/vid2bp/config/parameter.json") as f:
    json_data = json.load(f)
    param = json_data.get("parameters")
    sr = json_data.get("parameters").get("sampling_rate")
    # chunk_size = json_data.get("parameters").get("chunk_size")


def sig_spliter(in_sig):
    if np.ndim(in_sig) == 2:
        # for mimicdataset ( 2 channels : [ABP, PPG] )
        if np.shape(in_sig)[-1] == 2:
            abp_split, ple_split = np.split(in_sig, 2, axis=1)
        # for ucidataset ( 3 channels : [PPG, ABP, ECG] / ECG not needed )
        else:
            ple_split, abp_split, _ = np.split(in_sig, 3, axis=1)
        return abp_split, ple_split

    elif np.ndim(in_sig) == 3:  # ndim==3
        if np.shape(in_sig)[-1] == 2:
            abp_split, ple_split = np.split(in_sig, 2, axis=2)
            return abp_split, ple_split
        else:
            print("not supported shape for sig_spliter() due to different length of data")

    else:
        print("not supported dimension for sig_spliter()")


'''
can be done either before or after sig_spliter()
'''


def down_sampling(in_signal, sampling_rate=60):
    # base sampling rate : 125Hz
    if sampling_rate in [30, 60, 120]:
        rst_sig = signal.resample(in_signal, int(param["chunk_size"] / sr["base"]) * sr[str(sampling_rate)])
        return rst_sig
    else:
        print("not supported sampling rate.. check parameter.json")
        return None


def scaler(input_sig):
    input_sig = np.reshape(input_sig, (-1, 1))
    scaled_output = preprocessing.MinMaxScaler(feature_range=(0, 3)).fit_transform(input_sig)
    return np.squeeze(scaled_output)


def peak_detection(in_signal):
    # TODO SBP, DBP 구해야 함  SBP : Done
    x = np.squeeze(in_signal)
    mean = np.mean(x)
    peaks, prop = signal.find_peaks(x, height=mean, distance=30)

    return peaks, prop["peak_heights"], len(peaks)


def get_systolic(input_sig):
    x = np.squeeze(input_sig)
    _, prop = signal.find_peaks(x, height=np.max(input_sig) - np.std(input_sig))
    sbp = prop["peak_heights"]
    return sbp


# TODO : make get_diastolic() function available for mimic dataset
def get_diastolic(input_sig):
    Fs = 60
    T = 1 / Fs

    s_fft = np.fft.fft(input_sig)
    amplitude = abs(s_fft) * (2 / len(s_fft))
    frequency = np.fft.fftfreq(len(s_fft), T)

    fft_freq = frequency.copy()
    peak_index = amplitude[:int(len(amplitude) / 2)].argsort()[-1]
    peak_freq = fft_freq[peak_index]
    if peak_freq == 0:
        peak_index = amplitude[:int(len(amplitude) / 2)].argsort()[-2]
        peak_freq = fft_freq[peak_index]

    cycle_len = round(Fs / peak_freq)

    dbp = []
    for i, idx in enumerate(range(int(len(input_sig) / cycle_len))):
        cycle_min = np.min(input_sig[i * cycle_len:(i + 1) * cycle_len])
        dbp.append(cycle_min)
    return dbp


def signal_quality_checker(input_sig, is_abp):
    s = get_systolic(input_sig)
    d = get_diastolic(input_sig)

    # print('sig_quality_checker', systolic_n, diastolic_n)
    if is_abp:
        if (np.abs(len(s) - len(d)) > 2) or (np.std(s) > 5) or (np.std(d) > 5) or (
                np.abs(np.mean(s) - np.mean(d)) < 20):
            return False
        else:
            return True
    else:
        if (np.abs(len(s) - len(d)) > 2) or (np.std(s) > 0.5) or (np.std(d) > 0.5):
            return False
        else:
            return True


def get_mean_blood_pressure(dbp, sbp):
    # mbp = []
    # for d, s in zip(dbp, sbp):
    #     mbp.append((2 * d + s) / 3)
    mbp = (2 * dbp + sbp) / 3
    return mbp


def frequency_checker(input_sig):
    '''
    https://lifelong-education-dr-kim.tistory.com/4
    '''
    Fs = 125
    T = 1 / Fs

    abnormal_cnt = 0
    flag = False

    s_fft = np.fft.fft(input_sig)
    amplitude = abs(s_fft) * (2 / len(s_fft))
    frequency = np.fft.fftfreq(len(s_fft), T)

    fft_freq = frequency.copy()
    dicrotic_peak_index0 = amplitude[:int(len(amplitude) / 2)].argsort()[-1]
    peak_freq = fft_freq[dicrotic_peak_index0]
    if peak_freq == 0:
        peak_index = amplitude[:int(len(amplitude) / 2)].argsort()[-2]
        peak_freq = fft_freq[peak_index]

    cycle_len = round(Fs / peak_freq)

    for i in range(int(len(input_sig) / cycle_len)):
        if i == 0:
            cycle = input_sig[i * cycle_len:(i + 1) * cycle_len]
        else:
            cycle = input_sig[(i - 1) * cycle_len:i * cycle_len]
        corr = r(cycle, input_sig[i * cycle_len:(i + 1) * cycle_len])
        if corr < 0.7:
            abnormal_cnt += 1
    if abnormal_cnt > 1:
        flag = True

    return flag


def chebyshev2(input_sig, low, high, sr):
    nyq = 0.5 * sr
    if high / nyq < 1:
        if high * 2 > 125:
            sos = signal.cheby2(4, 30, [low / nyq, high / nyq], btype='bandpass', output='sos')
        else:
            sos = signal.cheby2(4, 30, low / nyq, btype='highpass', output='sos')
        filtered = signal.sosfilt(sos, input_sig)
        return filtered
    else:
        print("wrong bandwidth.. ")


def signal_slicing(model_name, rawdata, chunk_size, sampling_rate, fft=True):
    abp_list = []
    ple_list = []
    size_list = []

    if np.shape(rawdata[0]) != (chunk_size, 2):
        print(np.shape(rawdata))
        print('current shape is not the way intended. please check UCIdataset.py')
        rawdata = np.reshape(rawdata, (-1, chunk_size, 2))
        print(np.shape(rawdata))
    cnt = 0
    abnormal_cnt = 0
    for data in tqdm(rawdata):
        abp, ple = sig_spliter(data)
        abp = np.squeeze(abp)
        ple = np.squeeze(ple)
        p_abp, pro_abp = signal.find_peaks(abp, height=np.max(abp) - np.std(abp))
        p_ple, pro_ple = signal.find_peaks(ple, height=np.mean(ple))

        if not ((np.mean(ple) == 0.0) or
                (np.mean(abp) == 80.0) or
                (len(p_abp) < 3) or
                (len(p_ple) < 3) or
                (len(p_abp) - len(p_ple) > 1) or
                (signal_quality_checker(abp, is_abp=True) is False) or
                (signal_quality_checker(ple, is_abp=False) is False)):
            if fft is True:
                if (not frequency_checker(abp)) and (not frequency_checker(ple)):
                    abp = down_sampling(abp, sampling_rate)
                    ple = down_sampling(ple, sampling_rate)
                    abp_list.append(abp)
                    ple_list.append(ple)
                    if model_name == "BPNet":
                        size_list.append([
                            np.mean(get_diastolic(ple)),
                            np.mean(get_systolic(ple)),
                            get_mean_blood_pressure(np.mean(get_diastolic(ple)),
                                                    np.mean(get_systolic(ple)))])
                        cnt += 1
                    elif model_name == "Unet":
                        cnt += 1
                else:
                    abnormal_cnt += 1

            else:
                abp = down_sampling(abp, sampling_rate)
                ple = down_sampling(ple, sampling_rate)
                abp_list.append(abp)
                ple_list.append(ple)
                if model_name == "BPNet":
                    size_list.append([
                        np.mean(get_diastolic(ple)),
                        np.mean(get_systolic(ple)),
                        get_mean_blood_pressure(np.mean(get_diastolic(ple)),
                                                np.mean(get_systolic(ple)))])
                    cnt += 1
                elif model_name == "Unet":
                    cnt += 1

    print('measuring problem data slices : ', abnormal_cnt)
    print('passed :', cnt, '/ total :', len(rawdata))

    if model_name == "BPNet":
        return abp_list, ple_list, size_list
    elif model_name == "Unet":
        return abp_list, ple_list
