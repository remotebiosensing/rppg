import numpy as np
from scipy import signal
from sklearn import preprocessing
import matplotlib.pyplot as plt
# from vid2bp.nets.loss.loss import r
from tqdm import tqdm
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import json


def r(predictions, targets):
    # 대체로 0.3 이상이면 상관관계가 존재한다고 평가한다
    x_bar = (1 / len(predictions)) * np.sum(predictions)
    # print('x_bar :', x_bar)
    y_bar = (1 / len(targets)) * np.sum(targets)
    # print('y_bar :', y_bar)
    Sxx = 0
    Syy = 0
    Sxy = 0
    for x, y in zip(predictions, targets):
        Sxx += pow(x - x_bar, 2)
        Syy += pow(y - y_bar, 2)
        Sxy += (x - x_bar) * (y - y_bar)

    return Sxy / (np.sqrt(Sxx) * np.sqrt(Syy))


with open("/home/paperc/PycharmProjects/Pytorch_rppgs/vid2bp/config/parameter.json") as f:
    json_data = json.load(f)
    param = json_data.get("parameters")
    sr = json_data.get("parameters").get("sampling_rate")
    # chunk_size = json_data.get("parameters").get("chunk_size")

def channel_spliter(multi_sig):
    if np.ndim(multi_sig) == 2:
        # for mimicdataset ( 2 channels : [ABP, PPG] )
        if np.shape(multi_sig)[-1] == 2:
            abp_split, ple_split = np.split(multi_sig, 2, axis=1)
        # for ucidataset ( 3 channels : [PPG, ABP, ECG] / ECG not needed )
        else:
            ple_split, abp_split, _ = np.split(multi_sig, 3, axis=1)
        return np.squeeze(abp_split), np.squeeze(ple_split)

    elif np.ndim(multi_sig) == 3:  # ndim==3
        if np.shape(multi_sig)[-1] == 2:
            abp_split, ple_split = np.split(multi_sig, 2, axis=2)
            return np.squeeze(abp_split), np.squeeze(ple_split)
        else:
            print("not supported shape for sig_spliter() due to different length of data")

    else:
        print("not supported dimension for sig_spliter()")

class SignalHandler:
    def __init__(self, single_sig):
        self.single_sig = single_sig
        # self.multi_sig = multi_sig
        self.min = np.min(self.single_sig)
        self.max = np.max(self.single_sig)

    def down_sampling(self, sampling_rate=60):
        # base sampling rate : 125Hz
        if sampling_rate in [30, 60, 120]:
            rst_sig = signal.resample(self.single_sig,
                                      num=int(param["chunk_size"] / sr["base"]) * sr[str(sampling_rate)])
            return rst_sig
        else:
            print("not supported sampling rate.. check parameter.json")
            return None

    def scaler(self, min_val=min, max_val=max):
        input_sig = np.reshape(self.single_sig, (-1, 1))
        scaled_output = preprocessing.MinMaxScaler(feature_range=(min_val, max_val)).fit_transform(input_sig)
        return np.squeeze(scaled_output)

    def DC_value_removal(self):
        return self.single_sig - np.mean(self.single_sig)


def peak_detection(in_signal):
    # TODO SBP, DBP 구해야 함  SBP : Done
    x = np.squeeze(in_signal)
    mean = np.mean(x)
    peaks, prop = signal.find_peaks(x, height=mean, distance=30)

    return peaks, prop["peak_heights"], len(peaks)


class BPInfoExtractor:
    def __init__(self, input_sig):
        super().__init__()
        self.input_sig = input_sig
        self.sbp = self.get_systolic()
        self.dbp = self.get_diastolic()
        self.map = self.get_mean_arterial_pressure()

    def get_systolic(self):
        x = np.squeeze(self.input_sig)
        idx, prop = signal.find_peaks(x, height=np.max(self.input_sig) - np.std(self.input_sig))
        # idx, prop = signal.find_peaks(x, height=np.mean(input_sig))
        sbp = prop["peak_heights"]
        return [idx, sbp]

    # TODO : make get_diastolic() function available for mimic dataset
    def get_diastolic(self):
        cycle_len = get_cycle_len(self.input_sig)

        dbp = []
        for i, idx in enumerate(range(int(len(self.input_sig) / cycle_len))):
            cycle_min = np.min(self.input_sig[i * cycle_len:(i + 1) * cycle_len])
            dbp.append(cycle_min)
        return dbp

    def get_mean_arterial_pressure(self):
        # mbp = []
        # for d, s in zip(dbp, sbp):
        #     mbp.append((2 * d + s) / 3)
        mbp = (2 * np.mean(self.dbp) + np.mean(self.sbp[-1])) / 3
        return mbp


def signal_quality_checker(input_sig, is_abp):
    bp = BPInfoExtractor(input_sig)
    s = bp.get_systolic()[-1]
    d = bp.get_diastolic()
    # s = bp.get_systolic(input_sig)[-1]
    # d = bp.get_diastolic(input_sig)

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





def frequency_checker(input_sig):
    '''
    https://lifelong-education-dr-kim.tistory.com/4
    '''
    flag = False
    abnormal_cnt = 0
    cycle_len = get_cycle_len(input_sig)
    # print('-----------------')
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


def get_cycle_len(input_sig):
    Fs = 125
    T = 1 / Fs
    # DC_removed_signal = DC_value_removal(input_sig)
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

    return cycle_len


# def type_compare(new_type, exist_type):
#     if len(new_type) < 30:
#         return 1.0
#     else:
#         # new_type = signal.resample(new_type, len(exist_type)).tolist()
#         new_type = signal.resample(new_type, num=len(exist_type)).tolist()
#         # to make two lists in the same length using resample
#         # new_type = signal.resample_poly(new_type, len(exist_type), len(new_type)).tolist()
#
#         if len(new_type) > len(exist_type):
#             new_type = new_type[:len(exist_type)]
#         else:
#             exist_type = exist_type[:len(new_type)]
#         new_type = scaler(new_type, np.min(exist_type), np.max(exist_type))
#         corr = r(new_type, exist_type)
#         if corr < 0.5:
#             plt.plot(exist_type, 'r', label='exist')
#             plt.plot(new_type, 'b', label='new')
#             plt.legend()
#             plt.show()
#         else:
#             pass
#         return corr


def get_single_cycle(input_sig):
    idx = 2
    bp = BPInfoExtractor(input_sig)
    sys_list = bp.get_systolic(input_sig)[0]
    start_index = sys_list[idx]
    cycle_len = sys_list[idx + 1] - sys_list[idx]
    single_cycle = input_sig[start_index:start_index + cycle_len]
    if cycle_len < 10 or (np.max(single_cycle) - np.min(single_cycle)) < 10:
        idx += 1
        for i in range(3):
            sys_list = bp.get_systolic(input_sig)[0]
            start_index = sys_list[idx]
            cycle_len = sys_list[idx + 1] - sys_list[idx]
            single_cycle = input_sig[start_index:start_index + cycle_len]
            if cycle_len < 10 or (np.max(single_cycle) - np.min(single_cycle)) < 10:
                idx += 1
            else:
                break
    return single_cycle


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
    signal_type = []
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
        abp, ple = channel_spliter(data)
        p_abp, pro_abp = signal.find_peaks(abp, height=np.max(abp) - np.std(abp))
        p_ple, pro_ple = signal.find_peaks(ple, height=np.mean(ple))

        if not ((np.mean(ple) == (0.0 or np.nan)) or
                (np.mean(abp) == 80.0) or
                (len(p_abp) < 5) or
                (len(p_ple) < 5) or
                (len(p_abp) - len(p_ple) > 1) or
                (signal_quality_checker(abp, is_abp=True) is False) or

                (signal_quality_checker(ple, is_abp=False) is False)):
            abp = SignalHandler(abp).down_sampling(sampling_rate)
            ple = SignalHandler(ple).down_sampling(sampling_rate)
            if fft is True:
                if (not frequency_checker(abp)) and (not frequency_checker(ple)):
                    '''
                    signal type classification
                    '''
                    # new_type = get_single_cycle(abp)
                    # # 처음 type은 무조건 signal type에 추가
                    # if type_flag == False:
                    #     type_flag = True
                    #     plt.title('first type')
                    #     signal_type.append(new_type)
                    #     plt.plot(new_type)
                    #     plt.show()
                    #     type_cnt += 1
                    #     print('new signal type added!! :', len(signal_type))
                    # # signal_type에 있는 type들과 새로 들어온 type과의 correlation을 비교
                    # else:
                    #     # corr = type_compare(new_type, signal_type[-1])
                    #     # if corr < 0.5:
                    #     #     signal_type.append(new_type)
                    #     #     print('new signal type added!! :', len(signal_type))
                    #     for t in reversed(signal_type):
                    #         corr = type_compare(new_type, t)
                    #         if corr < 0.5:
                    #             signal_type.append(new_type)
                    #             print('new signal type added!! :', len(signal_type))
                    #             break
                    #         else:
                    #             continue

                    abp = SignalHandler(abp).down_sampling(sampling_rate)
                    ple = SignalHandler(ple).down_sampling(sampling_rate)
                    # ple = down_sampling(ple, sampling_rate)
                    abp_list.append(abp)
                    ple_list.append(ple)
                    if model_name == "BPNet":
                        bp = BPInfoExtractor(abp)
                        dia, sys = bp.dbp, bp.sbp[-1]
                        size_list.append([np.mean(dia), np.mean(sys), bp.get_mean_arterial_pressure])
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
                    bp = BPInfoExtractor(abp)
                    dia, sys = bp.dbp, bp.sbp[-1]
                    size_list.append([np.mean(dia), np.mean(sys), bp.get_mean_arterial_pressure])
                    cnt += 1
                elif model_name == "Unet":
                    cnt += 1

    print('measuring problem data slices : ', abnormal_cnt)
    print('passed :', cnt, '/ total :', len(rawdata))
    print('-----type of signal----- : ', len(signal_type))

    # for i in range(len(signal_type)):
    #     plt.plot(signal.resample(scaler(signal_type[i], 60, 120), num=100))
    # plt.show()
    if model_name == "BPNet":
        return abp_list, ple_list, size_list
    elif model_name == "Unet":
        return abp_list, ple_list
