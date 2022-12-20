import numpy as np
from scipy import signal
from sklearn import preprocessing
import matplotlib.pyplot as plt
# from vid2bp.nets.loss.loss import r
from tqdm import tqdm
import pandas as pd
# from statsmodels.tsa.seasonal import seasonal_decompose
import json
from heartpy.filtering import filter_signal
import heartpy.peakdetection as hp_peak
from heartpy.datautils import rolling_mean


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


with open("/home/najy/PycharmProjects/rppgs/vid2bp/config/parameter.json") as f:
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

def signal_respiration_checker(ABP, PPG, threshold=0.9):
    ABP = filter_signal(np.squeeze(ABP), cutoff=3, sample_rate=125., order=2, filtertype='lowpass')
    PPG = filter_signal(np.squeeze(PPG), cutoff=3, sample_rate=125., order=2, filtertype='lowpass')

    # Normalization
    ABP = 2 * (ABP - np.min(ABP)) / (np.max(ABP) - np.min(ABP)) - 1
    PPG = 2 * (PPG - np.min(PPG)) / (np.max(PPG) - np.min(PPG)) - 1
    # Peak detection
    rolling_sec = 0.75
    r_rolling_sec = 0.5
    SBP = SBP_detection(ABP, rolling_sec)
    DBP = DBP_detection(ABP, rolling_sec)
    SBP, DBP = SBP_DBP_filter(ABP, SBP, DBP)
    PPG_peak = SBP_detection(PPG, rolling_sec)
    PPG_low = DBP_detection(PPG, rolling_sec)
    PPG_peak, PPG_low = SBP_DBP_filter(PPG, PPG_peak, PPG_low)
    # Matching peaks
    matched_ABP, matched_PPG, gap_size, SBP, DBP, PPG_peak, PPG_low = match_signal(ABP, PPG, SBP, DBP,
                                                                                   PPG_peak, PPG_low)
    # ABP, PPG Rolling mean
    ABP_rolling_mean, PPG_rolling_mean = signals_rolling_mean(matched_ABP, matched_PPG, r_rolling_sec)

    # Normalization
    ABP_rolling_mean = 2 * (ABP_rolling_mean - np.min(ABP_rolling_mean)) / (
            np.max(ABP_rolling_mean) - np.min(ABP_rolling_mean)) - 1
    PPG_rolling_mean = 2 * (PPG_rolling_mean - np.min(PPG_rolling_mean)) / (
            np.max(PPG_rolling_mean) - np.min(PPG_rolling_mean)) - 1

    # correlation @rolling mean
    correlation = np.mean(np.corrcoef(ABP_rolling_mean, PPG_rolling_mean))
    if correlation >= threshold:
        return True
    else:
        return False

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
