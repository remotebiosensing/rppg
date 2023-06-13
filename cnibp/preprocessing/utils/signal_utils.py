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
        # self.sig3 = sig3

    def shuffle_lists(self):
        '''
        shuffles two lists in same order
        '''
        sig1_chunks = SignalHandler(self.sig1).list_slice(self.chunk_size)
        sig2_chunks = SignalHandler(self.sig2).list_slice(self.chunk_size)
        # sig3_chunks = SignalHandler(self.sig3).list_slice(self.chunk_size)
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

    def normalize(self):
        min = np.min(self.input_sig)
        max = np.max(self.input_sig)
        return (self.input_sig - min) / (max - min)
    def zscorenorm(self):
        return (self.input_sig - np.mean(self.input_sig)) / np.std(self.input_sig)
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

    def __init__(self, input_sig, normalize, preprocessing_mode):
        super().__init__(input_sig)
        # self.input_sig = SignalHandler(input_sig).savitzky_golay_filter(15, 2, 'nearest')
        # self.input_sig = SignalHandler.savitzky_golay_filter(input_sig)
        # self.ple_flag = np.max(input_sig) < 10
        if normalize:
            # self.input_sig = SignalHandler(input_sig).normalize()
            plt.plot(input_sig)
            self.input_sig = SignalHandler(input_sig).zscorenorm()
            plt.plot(self.input_sig)
            plt.show()
            print('ppg')
        else:
            self.input_sig = input_sig
        self.mode = preprocessing_mode
        self.fs = 125
        self.freq_flag, self.cycle_len, self.fft_bpm = self.get_cycle_len()
        self.rolling_sec = self.cycle_len / self.fs  # rolling sec is decided via cycle detection

        self.sbp_flag, self.sbp_idx, self.sbp_value = self.get_systolic()
        self.dbp_flag, self.dbp_idx, self.dbp_value = self.get_diastolic()
        # self.notch = self.get_dicrotic_notch()
        self.cycle_flag, self.cycle = self.get_cycle()

        if 'flat' in self.mode:
            self.flat_flag = self.flat_detection()
        self.status = self.return_sig_status()
        self.valid_flag = self.signal_validation()  # check if signal is valid with sbp, dbp, flip, flat flags

    def get_mahalanobis_dis(self, x):
        """
        to check mahalanobis distance of detected peaks intervals and values
        """
        x = np.diff(x)

        maha_dis = np.abs(x - np.mean(x)) / np.std(x)
        abnormal_checker = np.where(maha_dis > 2)
        if len(abnormal_checker[0]) > 0:
            return False
        else:
            return True

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
            if not self.get_mahalanobis_dis(cycle_len_list):
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
        if self.cycle_len < 200:
            diff_check = np.where(np.diff(sbp_idx) <= self.cycle_len * 0.5)
        else:
            diff_check = np.where(np.diff(sbp_idx) <= np.mean(np.diff(sbp_idx)) * 0.5)
        if len(diff_check[0]) > 0:
            sbp_idx = np.delete(sbp_idx, diff_check[0] + 1)

        if len(sbp_idx) < len(self.input_sig) // self.cycle_len - 1:
            #     ''' meaning that there are not enough DBP detected compared to the length of the signal '''
            return False, sbp_idx, self.input_sig[sbp_idx]
        elif not self.get_mahalanobis_dis(sbp_idx) or not self.get_mahalanobis_dis(self.input_sig[sbp_idx]):
            ''' meaning that the intervals between SBP are not consistent'''
            return False, sbp_idx, self.input_sig[sbp_idx]
        else:
            return True, sbp_idx, self.input_sig[sbp_idx]

    def get_diastolic(self):
        inverted_sig = -self.input_sig
        roll_mean = rolling_mean(inverted_sig, self.rolling_sec, self.fs)
        roll_mean2 = rolling_mean(inverted_sig, self.rolling_sec * 0.8, self.fs)
        dbp_idx = hp_peak.detect_peaks(inverted_sig, roll_mean, ma_perc=20, sample_rate=self.fs)['peaklist']
        dbp_idx2 = hp_peak.detect_peaks(inverted_sig, roll_mean2, ma_perc=20, sample_rate=self.fs)['peaklist']

        if np.std(np.diff(dbp_idx)) > np.std(np.diff(dbp_idx2)):
            dbp_idx = dbp_idx2
        if self.cycle_len < 200:
            diff_check = np.where(np.diff(dbp_idx) <= self.cycle_len * 0.5)
        else:
            diff_check = np.where(np.diff(dbp_idx) <= np.mean(np.diff(dbp_idx)) * 0.5)
        if len(diff_check[0]) > 0:
            dbp_idx = np.delete(dbp_idx, diff_check[0] + 1)
        if len(dbp_idx) < len(self.input_sig) // self.cycle_len - 1:
            #     ''' meaning that there are not enough SBP detected compared to the length of the signal '''
            return False, dbp_idx, self.input_sig[dbp_idx]
        elif not self.get_mahalanobis_dis(dbp_idx) or not self.get_mahalanobis_dis(self.input_sig[dbp_idx]):
            ''' meaning that the intervals between DBP is not consistent'''
            return False, dbp_idx, self.input_sig[dbp_idx]
        else:
            return True, dbp_idx, self.input_sig[dbp_idx]

    def get_dicrotic_notch(self):
        if self.dbp_flag is True and self.sbp_flag is True:
            sb_list = sorted(np.append(self.sbp_flag, self.dbp_flag))
            if self.input_sig[sb_list[0]] > self.input_sig[sb_list[1]]:
                start_idx, end_idx = sb_list[0], sb_list[1]
            else:
                start_idx, end_idx = sb_list[1], sb_list[2]
            detect_range = self.input_sig[start_idx:end_idx]
            plt.plot(self.input_sig)
            plt.plot(np.arange(start_idx, end_idx), detect_range)
            plt.show()
            return True

    def flat_detection(self):
        flat_range = np.where(self.input_sig == np.max(self.cycle))[0]

        if len(flat_range) < len(self.input_sig) * 0.05:  # flat 한 부분이 5% 미만일 때
            return False
        else:
            return True

    def return_sig_status(self):
        sig_status = [self.dbp_flag, self.sbp_flag, self.cycle_flag]
        if 'flat' in self.mode:
            sig_status.append(self.flat_flag)

        return list(map(int, sig_status))

    def signal_validation(self):
        if sum(self.status[3:]) == 0 and sum(self.status[:3]) == 3:
            return True
        else:
            return False

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
    def __init__(self, input_sig, normalize, preprocessing_mode):
        super().__init__(input_sig, normalize, preprocessing_mode)
        self.mode = preprocessing_mode
        self.amp_flag = self.amp_checker()  # True 사용
        self.pulse_pressure_flag, self.pulse_pressure = self.pulse_pressure_checker()  # True 사용
        if 'flip' in self.mode:
            self.flip_flag = self.flip_detection()  # if flipped returns True
        if 'underdamp' in self.mode:
            self.under_damped_flag = self.under_damped_detection()  # True 사용
        # if 'overdamp' in self.mode:
        #     self.over_damped_flag = self.over_damped_detection()  # True 사용
        self.detail_status = self.return_abp_sig_status()  # True 사용
        self.abp_valid_flag = self.valid_flag and self.abp_signal_validation()  # True 사용

    def flip_detection(self):
        if np.argmax(self.cycle) > len(self.cycle) / 2:  # 가장 큰 값이 cycle의 중간보다 뒤에 있을 때 ( 비정상 )
            # plt.plot(self.cycle)
            # plt.show()
            return True

        else:  # 가장 큰 값이 cycle의 중간보다 앞에 있을 때 ( 정상 )
            temp = [0.01685, 0.01962, 0.03278, 0.05362, 0.08079, 0.11172, 0.14417, 0.17636, 0.20703, 0.23538,
                    0.26207, 0.28734, 0.31153, 0.33443, 0.35658, 0.37757, 0.39718, 0.41555, 0.43271, 0.44814,
                    0.46215, 0.47478, 0.48593, 0.49568, 0.50381, 0.51039, 0.51598, 0.52047, 0.52438, 0.52879,
                    0.53501, 0.54361, 0.55540, 0.57188, 0.59444, 0.62379, 0.65954, 0.70015, 0.74289, 0.78456,
                    0.82302, 0.85675, 0.88570, 0.91032, 0.93015, 0.94444, 0.95311, 0.95700, 0.95653, 0.95122,
                    0.94072, 0.92486, 0.90624, 0.88620, 0.86563, 0.84454, 0.82384, 0.80356, 0.78373, 0.76398,
                    0.74435, 0.72489, 0.70608, 0.68725, 0.66885, 0.65076, 0.63322, 0.61627, 0.59990, 0.58373,
                    0.56793, 0.55251, 0.53729, 0.52231, 0.50754, 0.49274, 0.47823, 0.46318, 0.44747, 0.43052,
                    0.41254, 0.39341, 0.37326, 0.35163, 0.32875, 0.30373, 0.27742, 0.25016, 0.22357, 0.19834,
                    0.17506, 0.15335, 0.13353, 0.11505, 0.09776, 0.08095, 0.06493, 0.04957, 0.03603, 0.02339]

            checker = np.corrcoef(self.cycle, signal.resample(temp, len(self.cycle)))[0, 1]
            if checker > 0.95:
                return True

            else:
                return False

    def amp_checker(self):
        if np.min(self.input_sig) < 30 or np.max(self.input_sig) > 240:  # 신호의 범위가 30~220 이 아니면
            return True
        else:  # 신호의 범위가 30~240 이면
            return False

    def pulse_pressure_checker(self):
        if self.sbp_flag and self.dbp_flag:
            if len(self.dbp_idx) >= len(self.sbp_idx):
                pp = self.sbp_value - self.dbp_value[:len(self.sbp_value)]
            elif len(self.dbp_idx) < len(self.sbp_idx):
                pp = self.sbp_value[:len(self.dbp_value)] - self.dbp_value
            else:
                pp = self.sbp_value - self.dbp_value
        else:
            return False, None
        if np.mean(pp) < np.mean(self.sbp_value) * 0.25:
            return True, pp
        elif np.mean(pp) > 100:
            return True, pp
        else:
            return False, pp

    def under_damped_detection(self):
        if self.cycle_flag:
            start_point = np.argmax(self.cycle)
            end_point = start_point + int(len(self.cycle) * 0.03)

            diff = np.diff(self.cycle[start_point:end_point])
            if np.mean(diff) < -5:  # 최고점에서 3 프레임의 기울기가 -5보다 작으면 ( 너무 급격히 떨어지면 )
                return True
            else:  # 최고점에서 3 프레임의 기울기가 -5보다 크면 ( 너무 급격히 떨어지지 않으면 )
                return False
        else:
            return False

    # TODO : over damped signal detection
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
        if 'flip' in self.mode:
            sig_status.append(self.flip_flag)
        if 'underdamp' in self.mode:
            sig_status.append(self.under_damped_flag)
        # if 'overdamp' in self.mode:
        #     sig_status.append(self.over_damped_flag)
        return list(map(int, sig_status))

    def abp_signal_validation(self):
        if sum(self.detail_status) == 0 and self.valid_flag:
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


def signal_comparator(ple, abp, ple_normalize, preprocessing_mode, threshold=0.9, sampling_rate=125):
    ple_info = SignalInfoExtractor(ple, ple_normalize, preprocessing_mode)
    abp_info = ABPSignalInfoExtractor(abp, not ple_normalize, preprocessing_mode)
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
