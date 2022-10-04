import numpy as np
from scipy import signal
from sklearn import preprocessing
import matplotlib.pyplot as plt
from nets.loss.loss import r

import json

with open("/home/paperc/PycharmProjects/VBPNet/config/parameter.json") as f:
    json_data = json.load(f)
    param = json_data.get("parameters")
    sr = json_data.get("parameters").get("sampling_rate")
    chunk_size = json_data.get("parameters").get("chunk_size")


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
        print("not supported sampling rate..")
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
    _, prop = signal.find_peaks(x, height=np.max(input_sig)-np.std(input_sig), distance=30)
    sbp = prop["peak_heights"]
    return sbp


def get_diastolic(input_sig):
    Fs = 60
    T = 1 / Fs

    s_fft = np.fft.fft(input_sig)
    amplitude = abs(s_fft) * (2 / len(s_fft))
    frequency = np.fft.fftfreq(len(s_fft), T)

    fft_freq = frequency.copy()
    dicrotic_peak_index0 = amplitude[:int(len(amplitude) / 2)].argsort()[-2]

    peak_freq = fft_freq[dicrotic_peak_index0]

    cycle_len = round(Fs / peak_freq)

    dbp = []
    for i, idx in enumerate(range(int(len(input_sig) / cycle_len))):
        cycle_min = np.min(input_sig[i * cycle_len:(i + 1) * cycle_len])
        dbp.append(cycle_min)
    return dbp


def signal_quality_checker(input_sig, is_abp):
    s = get_systolic(input_sig)
    d = get_diastolic(input_sig)
    s_std = np.std(s)
    d_std = np.std(d)
    systolic_n = len(s)
    diastolic_n = len(d)
    systolic_mean = np.mean(s)
    diastolic_mean = np.mean(d)
    if is_abp:
        if (np.abs(systolic_n - diastolic_n) > 2) or (s_std > 5) or (d_std > 5) or (np.abs(systolic_mean-diastolic_mean)<20):
            return False
        else:
            return True
    else:
        if (np.abs(systolic_n - diastolic_n) > 2) or (s_std > 0.5) or (d_std > 0.5):
            return False
        else:
            return True


def get_mean_blood_pressure(dbp, sbp):
    # mbp = []
    # for d, s in zip(dbp, sbp):
    #     mbp.append((2 * d + s) / 3)
    mbp = (2 * dbp + sbp) / 3
    return mbp


# TODO IF STD(PEAK) IS MUCH LARGER THAN PRIOR SLICE, THEN DROP


def frequency_checker(input_sig):
    '''
    https://lifelong-education-dr-kim.tistory.com/4
    '''
    Fs = 60
    T = 1 / Fs

    abnormal_cnt = 0
    flag = False

    s_fft = np.fft.fft(input_sig)
    amplitude = abs(s_fft) * (2 / len(s_fft))
    frequency = np.fft.fftfreq(len(s_fft), T)

    # # Dicrotic Notch amplify
    fft_freq = frequency.copy()
    dicrotic_peak_index0 = amplitude[:int(len(amplitude) / 2)].argsort()[-2]
    peak_freq = fft_freq[dicrotic_peak_index0]

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

    # plt.plot(filtered_data[:cycle], color='indigo')
    # plt.plot(input_sig)
    # plt.title("amplified signal")
    # plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.70)
    #
    # plt.show()
    # print("break point")
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


def signal_slicing(rawdata, sampling_rate, fft=True):
    abp_list = []
    ple_list = []
    size_list = []

    if np.shape(rawdata[0]) != (chunk_size, 2):
        rawdata = np.reshape(rawdata, (-1, chunk_size, 2))
    cnt = 0
    abnormal_cnt = 0
    from tqdm import tqdm
    if fft is True:
        for data in tqdm(rawdata):
            abp, ple = sig_spliter(data)
            abp = down_sampling(np.squeeze(abp), sampling_rate)
            ple = down_sampling(np.squeeze(ple), sampling_rate)

            p_abp, pro_abp = signal.find_peaks(abp, height=np.max(abp) - np.std(abp), distance=30)
            p_ple, pro_ple = signal.find_peaks(ple, height=np.mean(ple), distance=30)

            if not ((np.mean(ple) == (0.0 or np.nan)) or
                    (np.mean(abp) == 80.0) or
                    (len(p_abp) < 5) or
                    (len(p_ple) < 5) or
                    (len(p_abp) - len(p_ple) >= 1) or
                    (signal_quality_checker(abp, is_abp=True) is False) or
                    (signal_quality_checker(ple, False) is False)):
                if (not frequency_checker(abp)) and (not frequency_checker(ple)):
                    # abp_list.append(abp)
                    # ple_list.append(scaler(ple))
                    temp = np.empty(3)
                    temp[0] = np.mean(get_diastolic(abp))
                    temp[1] = np.mean(get_systolic(abp))
                    temp[2] = get_mean_blood_pressure(temp[0], temp[1])
                    '''AHA'''
                    # '''Normal'''
                    # if (temp[1] < 120) and (temp[0] < 80):
                    # '''Elevated'''
                    # if (120 <= temp[1] < 130) and (temp[0] < 80):
                    # '''Hypertension Stage 1'''
                    # if (130 <= temp[1] < 140) or (80 <= temp[0] < 90):
                    # '''Hypertension Stage 2'''
                    # if (140 <= temp[1]) or (90 <= temp[0]):
                    '''Hypertensive Crisis'''
                    if (180 <= temp[1]) or (120 <= temp[0]):
                        abp_list.append(abp)
                        ple_list.append(ple)
                        size_list.append(temp)
                        cnt += 1
                else:
                    abnormal_cnt += 1
                    # plt.plot(ple)
                    # plt.show()
        print('abnormal : ', abnormal_cnt)
        print(cnt, '/', len(rawdata))
    else:
        for data in tqdm(rawdata):
            abp, ple = sig_spliter(data)
            abp = down_sampling(np.squeeze(abp), sampling_rate)
            ple = down_sampling(np.squeeze(ple), sampling_rate)

            p_abp, pro_abp = signal.find_peaks(abp, height=np.max(abp) - np.std(abp), distance=30)
            pp_abp, ppro_abp = signal.find_peaks(abp, height=np.mean(abp) + np.std(abp))
            p_ple, pro_ple = signal.find_peaks(ple, height=np.mean(ple), distance=30)

            if not ((np.mean(ple) == (0.0 or np.nan)) or (np.mean(abp) == 80.0) or
                    (len(p_abp) < 7) or (len(p_ple) == 0) or
                    (len(p_abp) - len(p_ple) > 2) or (np.std(pro_abp["peak_heights"]) > 5) or
                    (np.mean(pro_abp["peak_heights"]) > 160) or (len(pp_abp) > 15)):
                cnt += 1
                # abp_list.append(abp)
                # ple_list.append(ple)
                temp = np.empty(3)
                temp[0] = get_diastolic(abp)
                temp[1] = get_systolic(abp)
                temp[2] = get_mean_blood_pressure(temp[0], temp[1])
                '''Normal'''
                if (temp[1] < 120) and (temp[0] < 80):
                    abp_list.append(abp)
                    ple_list.append(scaler(ple))
                    size_list.append(temp)
                # if 120 <= temp[1] < 130 and temp[0] < 80:             '''Elevated'''
                # if 130 <= temp[1] < 140 or 80 <= temp[0] < 90:        '''Hypertension Stage 1'''
                # if 140 <= temp[1] or 90 <= temp[0]:                   '''Hypertension Stage 2'''
                # if 180 <= temp[1] and 120 <= temp[0]:                 '''Hypertensive Crisis'''

        print(cnt, '/', len(rawdata))
    return abp_list, ple_list, size_list

# ''' train data load '''
# path = '/home/paperc/PycharmProjects/VBPNet/dataset/BPNet_mimic/raw.hdf5'
#
# # TODO use hdf5 file for training Done
# import h5py
# with h5py.File(path, "r") as f:
#     raw = np.array(f['raw'])
# abp, ple = sig_spliter(raw)
# test_abp = np.squeeze(abp[:750])


# d = get_diastolic_blood_pressure(test_abp)
# print(len(d), 'dbp list :', d)
# s = get_systolic_blood_pressure(test_abp)
# print(len(s), 'sbp list', s)
#
# m = get_mean_blood_pressure(d, s)
# print(len(m), 'mbp list', m)
