import numpy as np
from scipy import signal
from sklearn import preprocessing
import matplotlib.pyplot as plt
# from cnibp.nets.loss.loss import r
# from statsmodels.tsa.seasonal import seasonal_decompose
import json
import heartpy.peakdetection as hp_peak
from heartpy.datautils import rolling_mean
from cnibp.preprocessing.utils.signal_utils_base import SignalBase
import random

random.seed(125)

'''
class signal handler
    def down_sampling
    def scaler
    def 
'''


def select_mode(mode: str):
    """
    select what to consider during preprocessing
    """
    if mode == 'total':
        return ['flat', 'flip', 'underdamp', 'overdamp']
    elif mode == 'flat':
        return ['flat']
    elif mode == 'flip':
        return ['flip']
    elif mode == 'overdamp':
        return ['overdamp']
    elif mode == 'underdamp':
        return ['underdamp']
    elif mode == 'damp':
        return ['underdamp', 'overdamp']
    elif mode == 'noflat':
        return ['flip', 'underdamp', 'overdamp']
    elif mode == 'noflip':
        return ['flat', 'underdamp', 'overdamp']
    elif mode == 'nodamp':
        return ['flat', 'flip']
    elif mode == 'none':
        return []
    else:
        raise ValueError(
            'mode should be one of [total, flat, flip, underdamp, overdamp, noflat, noflip, nodamp, none]\n see doc in signal_utils.py')


class DualSignalHandler:
    def __init__(self, sig1, sig2, chunk_size=750):
        self.chunk_size = chunk_size
        self.sig1 = sig1
        self.sig2 = sig2

    def shuffle_lists(self):
        '''
        shuffles two lists in same order
        '''
        sig1_chunks = SignalHandler(self.sig1).list_slice(self.chunk_size)
        sig2_chunks = SignalHandler(self.sig2).list_slice(self.chunk_size)
        c = list(zip(sig1_chunks, sig2_chunks))
        random.shuffle(c)
        sig1_chunks, sig2_chunks = zip(*c)
        return sig1_chunks, sig2_chunks

    def stack_sigs(self, orient: str = 'vertical', resize_n: int = 15):
        '''
        vstack or hstack two lists
        '''
        if orient == 'vertical':
            return np.vstack((SignalHandler(self.sig1).resize_list(resize_n),
                              SignalHandler(self.sig2).resize_list(resize_n)))
        else:
            return np.hstack((SignalHandler(self.sig1).resize_list(resize_n),
                              SignalHandler(self.sig2).resize_list(resize_n)))
        # return np.stack((self.sig1, self.sig2), axis=1)



#
# def shuffle_two_list(a, b):
#     '''
#     shuffle two lists in same order
#     '''
#     c = list(zip(a, b))
#     random.shuffle(c)
#     a, b = zip(*c)
#     return list(a), list(b)


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


with open("/home/paperc/PycharmProjects/Pytorch_rppgs/cnibp/configs/parameter.json") as f:
    json_data = json.load(f)
    param = json_data.get("parameters")
    sr = json_data.get("parameters").get("sampling_rate")
    # chunk_size = json_data.get("parameters").get("chunk_size")


def get_mahalanobis_dis(x):
    """
    to check mahalanobis distance of detected peaks intervals and values
    """
    x = np.diff(x)
    # diff_check = np.where(x < 10)
    # x = np.delete(x, diff_check)
    mean = np.mean(x)
    std = np.std(x)
    maha_dis = np.abs(x - mean) / std
    abnormal_checker = np.where(maha_dis > 2)
    if len(abnormal_checker[0]) > 0:
        return False
    else:
        return True


class SignalHandler:
    def __init__(self, single_sig):
        self.input_sig = single_sig
        # self.multi_sig = multi_sig
        # self.min = np.min(self.input_sig)
        # self.max = np.max(self.input_sig)
        # self.filtered_sig = self.savitzky_golay_filter()

    def is_valid(self, threshold=0.1):
        if self.input_sig.std() < threshold or np.isnan(self.input_sig).any():
            return False
        else:
            return True

    def list_slice(self, n):
        # for i in range(0, len(lst), n):
        #     yield lst[i:i + n]
        return [self.input_sig[i:i + n] for i in range(0, len(self.input_sig), n)][:len(self.input_sig) // n]

    def down_sampling(self, sampling_rate=60):
        # base sampling rate : 125Hz
        if sampling_rate in [30, 60, 120]:
            rst_sig = signal.resample(self.input_sig,
                                      num=int(param["chunk_size"] / sr["base"]) * sr[str(sampling_rate)])
            return rst_sig
        else:
            print("not supported sampling rate.. check parameter.json")
            return None

    def scaler(self, min_val=min, max_val=max):
        input_sig = np.reshape(self.input_sig, (-1, 1))
        scaled_output = preprocessing.MinMaxScaler(feature_range=(min_val, max_val)).fit_transform(input_sig)
        return np.squeeze(scaled_output)

    def savitzky_golay_filter(self, window_size=21, poly_order=3, mode='nearest'):
        return signal.savgol_filter(self.input_sig, window_size, poly_order, mode=mode)

    def DC_value_removal(self):
        return self.input_sig - np.mean(self.input_sig)

    def resize_list(self, size=20):
        if len(self.input_sig) > size:
            return self.input_sig[:size]
        else:
            return np.append(self.input_sig, np.array([0] * (size - len(self.input_sig))))


class SignalInfoExtractor(SignalBase):
    '''
    * A class for extracting signal's morphological information
    * Both available for PPG and ABP Signal


    * For all detection functions, return True Flag for Valid Signal

    freq_flag : check if BPM is in range of 35 - 140
    dbp[0] : check if most of diastolic values are detected
    sbp[0] : check if most of systolic values are detected
    cycle_flag : check if enough cycles are detected
    flat_flag : check if signal has flat part
    flip_flag : check if signal is flipped for any reason
    valid_flag : all(above flags) should be True for valid signal
    '''

    def __init__(self, input_sig, preprocessing_mode):
        super().__init__(input_sig)
        # self.input_sig = SignalHandler(input_sig).savitzky_golay_filter(15, 2, 'nearest')
        # self.input_sig = SignalHandler.savitzky_golay_filter(input_sig)
        self.input_sig = input_sig
        self.mode = preprocessing_mode
        self.fs = 125
        self.freq_flag, self.cycle_len = self.get_cycle_len()
        self.rolling_sec = self.cycle_len / self.fs  # rolling sec is decided via cycle detection

        self.sbp_flag, self.sbp_idx, self.sbp_value, self.sbp_std = self.get_systolic()
        self.dbp_flag, self.dbp_idx, self.dbp_value, self.dbp_std = self.get_diastolic()
        # self.notch = self.get_dicrotic_notch()
        self.cycle_flag, self.cycle, self.feature_idx, self.delta_cycle = self.get_cycle()

        if 'flip' in self.mode:
            self.flip_flag = self.flip_detection()  # if flipped returns True
        if 'flat' in self.mode:
            self.flat_flag = self.flat_detection()
        self.status = self.return_sig_status()
        self.valid_flag = self.signal_validation()  # check if signal is valid with sbp, dbp, flip, flat flags

    def get_cycle_len(self):
        s_fft = np.fft.fft(self.input_sig)
        amplitude = (abs(s_fft) * (2 / len(s_fft)))[1:]
        frequency = np.fft.fftfreq(len(s_fft), 1 / self.fs)

        # fft_freq = frequency.copy()
        peak_index = amplitude[:int(len(amplitude) / 2)].argsort()[::-1][:2]
        peak_freq = frequency[peak_index[0]]
        if peak_freq <= 0:
            peak_freq = frequency[peak_index[1]]

        # fft_1x = s_fft.copy()
        # fft_1x[fft_freq != peak_freq] = 0
        # filtered_data = 2*np.fft.ifft(fft_1x)

        cycle_len = round(self.fs / peak_freq)
        bpm = peak_freq * 60
        if bpm > 140 or bpm < 35:
            # self.valid_flag = False
            return True, cycle_len, bpm
        else:
            return False, cycle_len, bpm

    # @property
    def get_cycle(self):
        cycle_list = []
        cycle_len_list = []
        lr_check_list = []
        if len(self.dbp_idx) >= 2:
            for i in range(len(self.dbp_idx) - 1):
                cycle = self.input_sig[self.dbp_idx[i]:self.dbp_idx[i + 1]]
                cycle_list.append(cycle)
                cycle_len_list.append(len(cycle))
                lr_check_list.append(abs(cycle[0] - cycle[-1]))
            avg_cycle_len = np.mean(cycle_len_list, dtype=np.int)
            # check if cycle lengths are similar
            if not get_mahalanobis_dis(cycle_len_list):
                return False, np.zeros(1)

            peak_bpm = (self.fs / avg_cycle_len) * 60

            if 35 < peak_bpm < 140 or 35 < self.fft_bpm < 140:
                length_order = np.argsort(np.abs(np.array(cycle_len_list) - avg_cycle_len))
                diff_order = np.argsort(lr_check_list)
                total_order = length_order[np.where((diff_order == length_order) == True)]
                if len(total_order) > 0:
                    most_promising_cycle_idx = total_order[0]
                else:
                    most_promising_cycle_idx = length_order[0]
                return True, cycle_list[most_promising_cycle_idx]
            else:
                return False, np.zeros(1)
        else:
            return False, np.zeros(1)

    def get_systolic(self):
        roll_mean = rolling_mean(self.input_sig, self.rolling_sec, self.fs)
        roll_mean2 = rolling_mean(self.input_sig, self.rolling_sec * 0.8, self.fs)
        sbp_idx = hp_peak.detect_peaks(self.input_sig, roll_mean, ma_perc=20, sample_rate=self.fs)['peaklist']
        sbp_idx2 = hp_peak.detect_peaks(self.input_sig, roll_mean2, ma_perc=20, sample_rate=self.fs)['peaklist']
        if np.std(np.diff(sbp_idx)) > np.std(np.diff(sbp_idx2)):
            sbp_idx = sbp_idx2
        if len(sbp_idx) != 0:
            temp_sig = self.input_sig[sbp_idx]
            temp_sig = (temp_sig - np.min(temp_sig)) / (np.max(temp_sig) - np.min(temp_sig))
            sbp_std = np.std(temp_sig)
        else:
            sbp_std = 10
        # sbp_val = self.input_sig[sbp_idx]
        if len(sbp_idx) <= 3 or len(sbp_idx) < len(self.input_sig) // np.mean(np.diff(sbp_idx)) - 1:
            ''' meaning that there are not enough DBP detected compared to the length of the signal '''
            return False, sbp_idx, self.input_sig[sbp_idx], sbp_std
        elif sbp_idx[-1] - sbp_idx[-2] < np.mean(np.diff(sbp_idx)) / 2:
            sbp_idx = np.delete(sbp_idx, -1)
        if np.std(np.diff(sbp_idx)) > np.mean(np.diff(sbp_idx)) * 0.2:
            ''' meaning that the intervals between SBP are not consistent'''
            return False, sbp_idx, self.input_sig[sbp_idx], sbp_std
        else:
            return True, sbp_idx, self.input_sig[sbp_idx], sbp_std

    def get_diastolic(self):
        inverted_sig = -self.input_sig
        roll_mean = rolling_mean(inverted_sig, self.rolling_sec, self.fs)
        roll_mean2 = rolling_mean(inverted_sig, self.rolling_sec * 0.8, self.fs)
        dbp_idx = hp_peak.detect_peaks(inverted_sig, roll_mean, ma_perc=20, sample_rate=self.fs)['peaklist']
        dbp_idx2 = hp_peak.detect_peaks(inverted_sig, roll_mean2, ma_perc=20, sample_rate=self.fs)['peaklist']
        if np.std(np.diff(dbp_idx)) > np.std(np.diff(dbp_idx2)):
            dbp_idx = dbp_idx2
        if len(dbp_idx) != 0:
            temp_sig = self.input_sig[dbp_idx]
            temp_sig = (temp_sig - np.min(temp_sig)) / (np.max(temp_sig) - np.min(temp_sig))
            dbp_std = np.std(temp_sig)
        else:
            dbp_std = 10
        # dbp_val = self.input_sig[dbp_idx]
        if len(dbp_idx) <= 3 or len(dbp_idx) < len(self.input_sig) // np.mean(np.diff(dbp_idx)) - 1:
            ''' meaning that there are not enough SBP detected compared to the length of the signal '''
            return False, dbp_idx, self.input_sig[dbp_idx], dbp_std
        elif dbp_idx[-1] - dbp_idx[-2] < np.mean(np.diff(dbp_idx)) / 2:
            dbp_idx = np.delete(dbp_idx, -1)
        if np.std(np.diff(dbp_idx)) > np.mean(np.diff(dbp_idx)) * 0.2:
            ''' meaning that the intervals between DBP is not consistent'''
            return False, dbp_idx, self.input_sig[dbp_idx], dbp_std
        else:

            return True, dbp_idx, self.input_sig[dbp_idx], dbp_std

    # def get_dicrotic_notch(self):
    #     if self.dbp[0] == True and self.sbp[0] == True:
    #         sb_list = sorted(np.append(self.sbp[1], self.dbp[1]))
    #         if self.input_sig[sb_list[0]] > self.input_sig[sb_list[1]]:
    #             start_idx, end_idx = sb_list[0], sb_list[1]
    #         else:
    #             start_idx, end_idx = sb_list[1], sb_list[2]
    #         detect_range = self.input_sig[start_idx:end_idx]
    #         plt.plot(self.input_sig)
    #         plt.plot(np.arange(start_idx, end_idx), detect_range)
    #         plt.show()
    #         return True

    def flat_detection(self):
        if self.cycle_flag is True:  # cycle detection 이 되었을 때
            flat_range = np.where(self.input_sig == np.max(self.cycle))[0]
        else:
            flat_range = np.where(self.input_sig == np.max(self.input_sig))[0]
        if len(flat_range) < len(self.input_sig) * 0.05:  # flat 한 부분이 5% 미만일 때
            return False
        else:
            return True

    def flip_detection(self):
        if self.cycle_flag is False:  # if cycle is not detected
            return True

        elif self.cycle_flag:  # cycle detection 이 되었을 때
            if np.argmax(self.cycle) > len(self.cycle) / 2:  # 가장 큰 값이 cycle의 중간보다 뒤에 있을 때 ( 비정상 )
                return True
            else:  # 가장 큰 값이 cycle의 중간보다 앞에 있을 때 ( 정상 )
                return False

    def return_sig_status(self):
        # sig_status = [self.dbp[0], self.sbp[0], self.freq_flag, self.flat_flag, self.flip_flag]
        sig_status = [self.dbp_flag, self.sbp_flag, self.freq_flag]
        if 'flip' in self.mode:
            sig_status.append(self.flip_flag)
        if 'flat' in self.mode:
            sig_status.append(self.flat_flag)

        return list(map(int, sig_status))

    def signal_validation(self):
        if sum(self.status[2:]) == 0:
            return True
        else:
            return False
        # if self.sbp_flag and self.dbp_flag and abs(len(self.sbp_idx) - len(self.dbp_idx)) < 2 and \
        #         not self.freq_flag and not self.flip_flag and not self.flat_flag:
        #     return True
        # else:
        #     return False

    def plot(self):
        status_dict = {0: 'dbp', 1: 'sbp', 2: 'freq', 3: 'flat', 4: 'flip',
                       5: 'out of range', 6: 'Pulse pressure', 7: 'under damped', 8: 'over damped'}
        status_list = self.status[2:]
        problem_list = []
        if sum(status_list) == 0:
            plt.title('Valid Signal')
        else:
            for i in range(len(status_list)):
                if status_list:
                    problem_list.append(status_dict[i])
            plt.title(problem_list)
        plt.plot(self.input_sig)
        plt.plot(self.sbp_idx, self.sbp_value, 'rx')
        plt.plot(self.dbp_idx, self.dbp_value, 'bx')
        try:
            plt.plot(self.feature_idx, self.input_sig[self.feature_idx], 'g.')
            plt.show()
            plt.close()
        except:
            plt.show()
            plt.close()


class ABPSignalInfoExtractor(SignalInfoExtractor):
    def __init__(self, input_sig, preprocessing_mode):
        super().__init__(input_sig, preprocessing_mode)
        self.mode = preprocessing_mode
        self.amp_flag = self.amp_checker()  # True 사용
        self.pulse_pressure_flag, self.pulse_pressure = self.pulse_pressure_checker()  # True 사용
        if 'damp' in self.mode:
            self.under_damped_flag = self.under_damped_detection()  # True 사용
        # if 'overdamp' in self.mode:
        #     self.over_damped_flag = self.over_damped_detection()  # True 사용
        self.detail_status = self.return_abp_sig_status()  # True 사용
        self.abp_valid_flag = self.valid_flag and self.abp_signal_validation()  # True 사용

    def amp_checker(self):
        if np.min(self.input_sig) < 30 or np.max(self.input_sig) > 240:  # 신호의 범위가 30~220 이 아니면
            return True
        else:  # 신호의 범위가 30~240 이면
            return False

    def pulse_pressure_checker(self):
        if self.sbp_flag and self.dbp_flag:
            if len(self.dbp_idx) > len(self.sbp_idx):
                pp = self.sbp_value - self.dbp_value[:len(self.sbp_value)]
            elif len(self.dbp_idx) < len(self.sbp_idx):
                pp = self.sbp_value[:len(self.dbp_value)] - self.dbp_value
            else:
                pp = self.sbp_value - self.dbp_value
        else:
            return True, None
        if np.mean(pp) < np.mean(self.sbp_value) * 0.25:
            return True, pp
        elif np.mean(pp) > 100:
            return True, pp
        else:
            return False, pp

    def under_damped_detection(self):
        # oscillation_cnt = np.diff()
        if self.cycle_flag is False:  # cycle detection이 안되었을 때
            return True
        elif self.cycle_flag:  # cycle detection이 되었을 때
            start_point = np.argmax(self.cycle)
            # end_point = start_point + 3
            end_point = start_point + int(len(self.cycle) * 0.03)

            diff = np.diff(self.cycle[start_point:end_point])
            if np.mean(diff) < -5:  # 최고점에서 3 프레임의 기울기가 -5보다 작으면 ( 너무 급격히 떨어지면 )
                return True
            else:  # 최고점에서 3 프레임의 기울기가 -5보다 크면 ( 너무 급격히 떨어지지 않으면 )
                return False

    def over_damped_detection(self):
        return False
        # if self.cycle_flag:
        #     if np.mean(self.sbp_value) < 120 and np.mean(self.dbp_value) > 80:
        #         # plt.title('over damped signal')
        #         # plt.plot(self.input_sig)
        #         # plt.plot(self.dbp_idx, self.dbp_value, 'bx')
        #         # plt.plot(self.sbp_idx, self.sbp_value, 'rx')
        #         # plt.show()
        #         return True
        #     else:
        #         return False
        # else:
        #     return False

    def return_abp_sig_status(self):
        sig_status = [self.amp_flag, self.pulse_pressure_flag]
        if 'damp' in self.mode:
            sig_status.append(self.under_damped_flag)
        # if 'overdamp' in self.mode:
        #     sig_status.append(self.over_damped_flag)
        return list(map(int, sig_status))

    def abp_signal_validation(self):
        if sum(self.detail_status) == 0:
            return True
        else:
            return False

    def plot(self):
        status_dict = {0: 'dbp', 1: 'sbp', 2: 'freq', 3: 'flat', 4: 'flip',
                       5: 'out of range', 6: 'Pulse pressure', 7: 'under damped', 8: 'over damped'}
        problem_list = []
        status_list = self.status[2:] + self.detail_status
        if sum(status_list) == 0:
            plt.title('Valid Signal')
        else:
            for i in range(len(status_list)):
                if status_list[i]:
                    problem_list.append(status_dict[i + 2])
            plt.title(problem_list)
        plt.plot(self.input_sig, label='ABP', color='black', linewidth=0.5)
        plt.plot(self.sbp_idx, self.sbp_value, 'rx', label='SBP')
        plt.plot(self.dbp_idx, self.dbp_value, 'bx', label='DBP')
        try:
            plt.plot(self.feature_idx, self.input_sig[self.feature_idx], 'g.', label='Feature point', markersize=5)
            plt.legend()
            plt.show()
            plt.close()
        except:
            plt.legend()
            plt.show()
            plt.close()


def signal_comparator(ple, abp, preprocessing_mode, threshold=0.9, sampling_rate=125):
    ple_info = SignalInfoExtractor(ple, preprocessing_mode)
    abp_info = ABPSignalInfoExtractor(abp, preprocessing_mode)
    if not ple_info.valid_flag:
        return False, ple_info, abp_info
    if not abp_info.abp_valid_flag:
        return False, ple_info, abp_info

    corr = np.corrcoef(signal.resample(ple_info.cycle, 100), signal.resample(abp_info.cycle, 100))[0, 1]
    # peak_diff = np.abs(len(ple_info.sbp[1]) - len(abp_info.sbp[1]))
    if ple_info.valid_flag and abp_info.abp_valid_flag:
        if corr > threshold:
            return True, ple_info, abp_info
        else:
            return False, ple_info, abp_info
    else:
        return False, ple_info, abp_info

# def signal_quality_checker(input_sig, is_abp):
#     bp = SignalInfoExtractor(input_sig)
#     s = bp.get_systolic()[-1]
#     d = bp.get_diastolic
#     # s = bp.get_systolic(input_sig)[-1]
#     # d = bp.get_diastolic(input_sig)
#
#     # print('sig_quality_checker', systolic_n, diastolic_n)
#     if is_abp:
#         if (np.abs(len(s) - len(d)) > 2) or (np.std(s) > 5) or (np.std(d) > 5) or (
#                 np.abs(np.mean(s) - np.mean(d)) < 20):
#             return False
#         else:
#             return True
#     else:
#         if (np.abs(len(s) - len(d)) > 2) or (np.std(s) > 0.5) or (np.std(d) > 0.5):
#             return False
#         else:
#             return True


# def channel_spliter(multi_sig):
#     if np.ndim(multi_sig) == 2:
#         # for mimicdataset ( 2 channels : [ABP, PPG] )
#         if np.shape(multi_sig)[-1] == 2:
#             abp_split, ple_split = np.split(multi_sig, 2, axis=1)
#         # for ucidataset ( 3 channels : [PPG, ABP, ECG] / ECG not needed )
#         else:
#             ple_split, abp_split, _ = np.split(multi_sig, 3, axis=1)
#         return np.squeeze(abp_split), np.squeeze(ple_split)
#
#     elif np.ndim(multi_sig) == 3:  # ndim==3
#         if np.shape(multi_sig)[-1] == 2:
#             abp_split, ple_split = np.split(multi_sig, 2, axis=2)
#             return np.squeeze(abp_split), np.squeeze(ple_split)
#         else:
#             print("not supported shape for sig_spliter() due to different length of data")
#
#     else:
#         print("not supported dimension for sig_spliter()")


# def ds_detection(ABP):
#     raw_ABP = ABP
#     ABP = filter_signal(np.squeeze(ABP), cutoff=3, sample_rate=60., order=2, filtertype='lowpass')
#     rolling_sec = 0.75
#     SBP = SBP_detection(ABP, rolling_sec)
#     DBP = DBP_detection(ABP, rolling_sec)
#     # SBP, DBP = SBP_DBP_filter(ABP, SBP, DBP)
#     if len(SBP) == 0 or len(DBP) == 0:
#         return 0, 0, 0, 0, 0
#         # return False, None, None, None
#     mean_sbp, mean_dbp = np.mean(raw_ABP[SBP]), np.mean(raw_ABP[DBP])
#     mean_map = (2 * mean_dbp + mean_sbp) / 3
#     return mean_sbp, mean_dbp, mean_map, SBP, DBP


# def signal_respiration_checker(ABP, PPG, threshold=0.9):
#     if np.isnan(PPG).any() or np.isnan(ABP).any() or \
#             (np.var(ABP) < 1) or \
#             (not (np.sign(ABP) > 0.0).all()):
#         return False, None, None, None
#     ABP = filter_signal(np.squeeze(ABP), cutoff=3, sample_rate=125., order=2, filtertype='lowpass')
#     PPG = filter_signal(np.squeeze(PPG), cutoff=3, sample_rate=125., order=2, filtertype='lowpass')
#
#     # Normalization
#     # Peak detection
#     rolling_sec = 0.75
#     r_rolling_sec = 0.5
#     SBP = SBP_detection(ABP, rolling_sec)
#     DBP = DBP_detection(ABP, rolling_sec)
#     SBP, DBP = SBP_DBP_filter(ABP, SBP, DBP)
#     if len(SBP) == 0 or len(DBP) == 0 or (abs(len(SBP) - len(DBP)) >= 2):
#         return False, None, None, None
#     mean_sbp, mean_dbp = np.mean(ABP[SBP]), np.mean(ABP[DBP])
#     mean_map = (2 * mean_dbp + mean_sbp) / 3
#     PPG_peak = SBP_detection(PPG, rolling_sec)
#     PPG_low = DBP_detection(PPG, rolling_sec)
#     PPG_peak, PPG_low = SBP_DBP_filter(PPG, PPG_peak, PPG_low)
#     ABP = 2 * (ABP - np.min(ABP)) / (np.max(ABP) - np.min(ABP)) - 1
#     PPG = 2 * (PPG - np.min(PPG)) / (np.max(PPG) - np.min(PPG)) - 1
#     # Matching peaks
#     if len(PPG_peak) == 0 or len(PPG_low) == 0 or len(SBP) == 0 or len(DBP) == 0:
#         return False, None, None, None
#     matched_ABP, matched_PPG, gap_size, SBP, DBP, PPG_peak, PPG_low = match_signal(ABP, PPG, SBP, DBP,
#                                                                                    PPG_peak, PPG_low)
#     # ABP, PPG Rolling mean
#     Flag, ABP_rolling_mean, PPG_rolling_mean = signals_rolling_mean(matched_ABP, matched_PPG, r_rolling_sec)
#
#     if Flag is False:
#         return False, None, None, None
#     # Normalization
#     ABP_rolling_mean = 2 * (ABP_rolling_mean - np.min(ABP_rolling_mean)) / (
#             np.max(ABP_rolling_mean) - np.min(ABP_rolling_mean)) - 1
#     PPG_rolling_mean = 2 * (PPG_rolling_mean - np.min(PPG_rolling_mean)) / (
#             np.max(PPG_rolling_mean) - np.min(PPG_rolling_mean)) - 1
#
#     # correlation @rolling mean
#     correlation = np.mean(np.corrcoef(ABP_rolling_mean, PPG_rolling_mean))
#     if correlation >= threshold:
#         return True, mean_dbp, mean_sbp, mean_map
#     else:
#         return False, None, None, None
#
#
# def window_wise_heartpy_peak_detection(signal, win_start, win_end, step=0.5, fs=125):
#     """
#     rolling mean():
#         windowsize : [sec], sample_rate : [Hz]
#
#     peak_hartpy():
#         ma_perc : the percentage with which to raise the rolling mean, used for fitting detection solutions to data
#     """
#     peaks = []
#     for window in np.arange(win_start, win_end, step=step):
#         rol_mean = rolling_mean(signal, window, fs)
#         peak_heartpy = hp_peak.detect_peaks(signal, rol_mean, ma_perc=20, sample_rate=fs)
#         peaks.append(peak_heartpy)
#     return peaks
#
#
# def SBP_detection(signal, rolling_sec=0.75, fs=125):
#     roll_mean = rolling_mean(signal, rolling_sec, fs)
#     peak_heartpy = hp_peak.detect_peaks(signal, roll_mean, ma_perc=20, sample_rate=fs)
#     return peak_heartpy['peaklist']
#
#
# def DBP_detection(signal, rolling_sec=0.75, fs=125):
#     signal = -signal
#     roll_mean = rolling_mean(signal, rolling_sec, fs)
#     peak_heartpy = hp_peak.detect_peaks(signal, roll_mean, ma_perc=20, sample_rate=fs)
#     return peak_heartpy['peaklist']
#
#
# def PPG_peak_detection(PPG, rolling_sec, fs=125):
#     PPG_rolling_mean = rolling_mean(PPG, rolling_sec, fs)
#     peak_heartpy = hp_peak.detect_peaks(PPG, PPG_rolling_mean, ma_perc=20, sample_rate=fs)
#     return peak_heartpy['peaklist']
#
#
# def match_signal(ABP, PPG, SBP, DBP, PPG_peak, PPG_low):
#     if PPG_peak[0] < SBP[0]:
#         matched_ABP = ABP[SBP[0]:]
#         matched_PPG, gap_size = PPG[PPG_peak[0]:len(matched_ABP) + PPG_peak[0]], PPG_peak[0] - SBP[0]
#     else:
#         matched_PPG = PPG[PPG_peak[0]:]
#         matched_ABP, gap_size = ABP[SBP[0]:len(matched_PPG) + SBP[0]], PPG_peak[0] - SBP[0]
#
#     if gap_size >= 0:
#         gap_size = SBP[0]
#         SBP = [SBP[x] - gap_size for x in range(len(SBP)) if
#                0 <= SBP[x] - gap_size < len(matched_ABP)]
#         DBP = [DBP[x] - gap_size for x in range(len(DBP)) if
#                0 <= DBP[x] - gap_size < len(matched_ABP)]
#         gap_size = PPG_peak[0]
#         PPG_peak = [PPG_peak[x] - gap_size for x in range(len(PPG_peak)) if
#                     0 <= PPG_peak[x] - gap_size < len(matched_PPG)]
#         PPG_low = [PPG_low[x] - gap_size for x in range(len(PPG_low)) if
#                    0 <= PPG_low[x] - gap_size < len(matched_PPG)]
#     else:
#         gap_size = PPG_peak[0]
#
#         PPG_peak = [PPG_peak[x] - gap_size for x in range(len(PPG_peak)) if
#                     len(matched_PPG) > PPG_peak[x] - gap_size >= 0]
#         PPG_low = [PPG_low[x] - gap_size for x in range(len(PPG_low)) if
#                    len(matched_PPG) > PPG_low[x] - gap_size >= 0]
#         gap_size = SBP[0]
#         SBP = [SBP[x] - gap_size for x in range(len(SBP)) if
#                len(matched_PPG) > SBP[x] - gap_size >= 0]
#         DBP = [DBP[x] - gap_size for x in range(len(DBP)) if
#                len(matched_PPG) > DBP[x] - gap_size >= 0]
#
#     return matched_ABP, matched_PPG, gap_size, SBP, DBP, PPG_peak, PPG_low
#
#
# def signals_rolling_mean(ABP, PPG, rolling_sec, fs=125):
#     # rolling mean for find proper trend
#     try:
#         ABP_rolling_mean = rolling_mean(ABP, rolling_sec, fs)
#         PPG_rolling_mean = rolling_mean(PPG, rolling_sec, fs)
#         return True, ABP_rolling_mean, PPG_rolling_mean
#     except:
#         return False, None, None
#
#
# def plot_signal_with_props(ABP, PPG, SBP, DBP, PPG_peak, PPG_low, ABP_rolling_mean, PPG_rolling_mean,
#                            title='signal with properties'):
#     plt.figure(figsize=(20, 5))
#     plt.plot(ABP)
#     plt.plot(PPG)
#     plt.plot(SBP, ABP[SBP], 'ro')
#     plt.plot(DBP, ABP[DBP], 'bo')
#     plt.plot(PPG_peak, PPG[PPG_peak], 'go')
#     plt.plot(PPG_low, PPG[PPG_low], 'yo')
#     plt.plot(ABP_rolling_mean, 'g', linestyle='--')
#     plt.plot(PPG_rolling_mean, 'y', linestyle='--')
#     plt.title(title)
#     plt.legend(['ABP', 'PPG', 'SBP', 'DBP', 'PPG_peak', 'PPG_low', 'ABP_rolling_mean', 'PPG_rolling_mean'])
#     plt.show()
#
#
# def SBP_DBP_filter(ABP, SBP, DBP):
#     i = 0
#     total = len(SBP) - 1
#     while i < total:
#         flag = False
#         # Distinguish SBP[i] < DBP < SBP[i+1]
#         for idx_dbp in DBP:
#             # Normal situation
#             if (SBP[i] < idx_dbp) and (idx_dbp < SBP[i + 1]):
#                 flag = True
#                 break
#             # abnormal situation
#         if flag:
#             i += 1
#         else:
#             # compare peak value
#             # delete smaller one @SBP
#             if ABP[SBP[i]] < ABP[SBP[i + 1]]:
#                 SBP = np.delete(SBP, i)
#             else:
#                 SBP = np.delete(SBP, i + 1)
#             total -= 1
#
#     i = 0
#     total = len(DBP) - 1
#     while i < total:
#         flag = False
#         # Distinguish DBP[i] < SBP < DBP[i+1]
#         for idx_sbp in SBP:
#             # Normal situation
#             if (DBP[i] < idx_sbp) and (idx_sbp < DBP[i + 1]):
#                 flag = True
#                 break
#         # normal situation
#         if flag:
#             i += 1
#         # abnormal situation, there is no SBP between DBP[i] and DBP[i+1]
#         else:
#             # compare peak value
#             # delete bigger one @DBP
#             if ABP[DBP[i]] < ABP[DBP[i + 1]]:
#                 DBP = np.delete(DBP, i + 1)
#             else:
#                 DBP = np.delete(DBP, i)
#             total -= 1
#
#     return SBP, DBP
#
#
# def peak_detection(in_signal):
#     # TODO SBP, DBP 구해야 함  SBP : Done
#     x = np.squeeze(in_signal)
#     mean = np.mean(x)
#     peaks, prop = signal.find_peaks(x, height=mean, distance=30)
#
#     return peaks, prop["peak_heights"], len(peaks)
#
#
# def frequency_checker(input_sig):
#     '''
#     https://lifelong-education-dr-kim.tistory.com/4
#     '''
#     flag = False
#     abnormal_cnt = 0
#     cycle_len = get_cycle_len(input_sig)
#     # print('-----------------')
#     for i in range(int(len(input_sig) / cycle_len)):
#         if i == 0:
#             cycle = input_sig[i * cycle_len:(i + 1) * cycle_len]
#         else:
#             cycle = input_sig[(i - 1) * cycle_len:i * cycle_len]
#         corr = r(cycle, input_sig[i * cycle_len:(i + 1) * cycle_len])
#         if corr < 0.7:
#             abnormal_cnt += 1
#     if abnormal_cnt > 1:
#         flag = True
#
#     return flag
#
#
# def get_cycle_len(input_sig):
#     Fs = 125
#     T = 1 / Fs
#     # DC_removed_signal = DC_value_removal(input_sig)
#     s_fft = np.fft.fft(input_sig)
#     amplitude = abs(s_fft) * (2 / len(s_fft))
#     frequency = np.fft.fftfreq(len(s_fft), T)
#
#     fft_freq = frequency.copy()
#     peak_index = amplitude[:int(len(amplitude) / 2)].argsort()[-1]
#     peak_freq = fft_freq[peak_index]
#     if peak_freq == 0:
#         peak_index = amplitude[:int(len(amplitude) / 2)].argsort()[-2]
#         peak_freq = fft_freq[peak_index]
#
#     cycle_len = round(Fs / peak_freq)
#
#     return cycle_len
#
#
# # def type_compare(new_type, exist_type):
# #     if len(new_type) < 30:
# #         return 1.0
# #     else:
# #         # new_type = signal.resample(new_type, len(exist_type)).tolist()
# #         new_type = signal.resample(new_type, num=len(exist_type)).tolist()
# #         # to make two lists in the same length using resample
# #         # new_type = signal.resample_poly(new_type, len(exist_type), len(new_type)).tolist()
# #
# #         if len(new_type) > len(exist_type):
# #             new_type = new_type[:len(exist_type)]
# #         else:
# #             exist_type = exist_type[:len(new_type)]
# #         new_type = scaler(new_type, np.min(exist_type), np.max(exist_type))
# #         corr = r(new_type, exist_type)
# #         if corr < 0.5:
# #             plt.plot(exist_type, 'r', label='exist')
# #             plt.plot(new_type, 'b', label='new')
# #             plt.legend()
# #             plt.show()
# #         else:
# #             pass
# #         return corr
#
#
# def get_single_cycle(input_sig):
#     idx = 2
#     bp = SignalInfoExtractor(input_sig)
#     sys_list = bp.get_systolic(input_sig)[0]
#     start_index = sys_list[idx]
#     cycle_len = sys_list[idx + 1] - sys_list[idx]
#     single_cycle = input_sig[start_index:start_index + cycle_len]
#     if cycle_len < 10 or (np.max(single_cycle) - np.min(single_cycle)) < 10:
#         idx += 1
#         for i in range(3):
#             sys_list = bp.get_systolic(input_sig)[0]
#             start_index = sys_list[idx]
#             cycle_len = sys_list[idx + 1] - sys_list[idx]
#             single_cycle = input_sig[start_index:start_index + cycle_len]
#             if cycle_len < 10 or (np.max(single_cycle) - np.min(single_cycle)) < 10:
#                 idx += 1
#             else:
#                 break
#     return single_cycle
#
#
# def chebyshev2(input_sig, low, high, sr):
#     nyq = 0.5 * sr
#     if high / nyq < 1:
#         if high * 2 > 125:
#             sos = signal.cheby2(4, 30, [low / nyq, high / nyq], btype='bandpass', output='sos')
#         else:
#             sos = signal.cheby2(4, 30, low / nyq, btype='highpass', output='sos')
#         filtered = signal.sosfilt(sos, input_sig)
#         return filtered
#     else:
#         print("wrong bandwidth.. ")
#
#
# def signal_slicing(model_name, rawdata, chunk_size, sampling_rate, fft=True):
#     signal_type = []
#     abp_list = []
#     ple_list = []
#     size_list = []
#
#     if np.shape(rawdata[0]) != (chunk_size, 2):
#         print(np.shape(rawdata))
#         print('current shape is not the way intended. please check UCIdataset.py')
#         rawdata = np.reshape(rawdata, (-1, chunk_size, 2))
#         print(np.shape(rawdata))
#     cnt = 0
#     abnormal_cnt = 0
#     for data in tqdm(rawdata):
#         abp, ple = channel_spliter(data)
#         p_abp, pro_abp = signal.find_peaks(abp, height=np.max(abp) - np.std(abp))
#         p_ple, pro_ple = signal.find_peaks(ple, height=np.mean(ple))
#
#         if not ((np.mean(ple) == (0.0 or np.nan)) or
#                 (np.mean(abp) == 80.0) or
#                 (len(p_abp) < 5) or
#                 (len(p_ple) < 5) or
#                 (len(p_abp) - len(p_ple) > 1) or
#                 (signal_quality_checker(abp, is_abp=True) is False) or
#                 (signal_quality_checker(ple, is_abp=False) is False)):
#             abp = SignalHandler(abp).down_sampling(sampling_rate)
#             ple = SignalHandler(ple).down_sampling(sampling_rate)
#             if fft is True:
#                 if (not frequency_checker(abp)) and (not frequency_checker(ple)):
#                     '''
#                     signal type classification
#                     '''
#                     # new_type = get_single_cycle(abp)
#                     # # 처음 type은 무조건 signal type에 추가
#                     # if type_flag == False:
#                     #     type_flag = True
#                     #     plt.title('first type')
#                     #     signal_type.append(new_type)
#                     #     plt.plot(new_type)
#                     #     plt.show()
#                     #     type_cnt += 1
#                     #     print('new signal type added!! :', len(signal_type))
#                     # # signal_type에 있는 type들과 새로 들어온 type과의 correlation을 비교
#                     # else:
#                     #     # corr = type_compare(new_type, signal_type[-1])
#                     #     # if corr < 0.5:
#                     #     #     signal_type.append(new_type)
#                     #     #     print('new signal type added!! :', len(signal_type))
#                     #     for t in reversed(signal_type):
#                     #         corr = type_compare(new_type, t)
#                     #         if corr < 0.5:
#                     #             signal_type.append(new_type)
#                     #             print('new signal type added!! :', len(signal_type))
#                     #             break
#                     #         else:
#                     #             continue
#
#                     abp = SignalHandler(abp).down_sampling(sampling_rate)
#                     ple = SignalHandler(ple).down_sampling(sampling_rate)
#                     # ple = down_sampling(ple, sampling_rate)
#                     abp_list.append(abp)
#                     ple_list.append(ple)
#                     if model_name == "BPNet":
#                         bp = SignalInfoExtractor(abp)
#                         dia, sys = bp.dbp, bp.sbp[-1]
#                         size_list.append([np.mean(dia), np.mean(sys), bp.get_mean_arterial_pressure])
#                         cnt += 1
#                     elif model_name == "Unet":
#                         cnt += 1
#                 else:
#                     abnormal_cnt += 1
#
#             else:
#                 abp = down_sampling(abp, sampling_rate)
#                 ple = down_sampling(ple, sampling_rate)
#                 abp_list.append(abp)
#                 ple_list.append(ple)
#                 if model_name == "BPNet":
#                     bp = SignalInfoExtractor(abp)
#                     dia, sys = bp.dbp, bp.sbp[-1]
#                     size_list.append([np.mean(dia), np.mean(sys), bp.get_mean_arterial_pressure])
#                     cnt += 1
#                 elif model_name == "Unet":
#                     cnt += 1
#
#     print('measuring problem data slices : ', abnormal_cnt)
#     print('passed :', cnt, '/ total :', len(rawdata))
#     print('-----type of signal----- : ', len(signal_type))
#
#     # for i in range(len(signal_type)):
#     #     plt.plot(signal.resample(scaler(signal_type[i], 60, 120), num=100))
#     # plt.show()
#     if model_name == "BPNet":
#         return abp_list, ple_list, size_list
#     elif model_name == "Unet":
#         return abp_list, ple_list
