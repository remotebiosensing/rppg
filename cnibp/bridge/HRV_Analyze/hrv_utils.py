import numpy as np
import torch
import matplotlib.pyplot as plt
from rppg.utils.HR_Analyze.UBFC_rppg import getPowerSpectrum
from scipy.signal import butter, lfilter, filtfilt
from scipy import signal
from rppg.utils.funcs import BPF, calc_hr_torch, detrend_torch
import pandas as pd
from tqdm import tqdm


class TimeHRVAnalysis:
    def __init__(self, input_signals: torch.tensor, fs: float = 30.):
        self.input_signals = input_signals  # self.remove_negative(input_signals)
        self.fs = fs

        self.hr = self.mean_hr()
        self.sdnn = self.calculate_sdnn()  # Standard Deviation of Interval
        self.rmssd = self.calculate_rmssd()
        self.nn50 = self.calculate_nn50()
        self.pnn50 = self.calculate_pnn50()
        # self.apen = self.approximate_entropy()
        self.srd = torch.ones_like(self.pnn50) - self.pnn50

    def mean_hr(self):
        length = torch.sum(self.input_signals > 0, dim=-1)
        masked_sig = self.input_signals + (self.input_signals < 0).to(torch.int)
        return 60 / ((torch.sum(masked_sig / 1000, dim=-1)) / length)

    def calculate_sdnn(self):
        # temp = self.input_signals
        sdnn = []
        for hrv in self.input_signals:
            sdnn.append(torch.std(hrv[hrv > 0]))

        return sdnn

    def calculate_rmssd(self):
        mean = torch.sum(torch.square(torch.diff(self.input_signals)), dim=-1) / torch.sum(self.input_signals > 0,
                                                                                           dim=-1)
        return torch.sqrt(mean)

    def calculate_nn50(self):
        return torch.sum(torch.diff(self.input_signals, dim=-1) >= 50, dim=-1)

    def calculate_pnn50(self):
        return self.nn50 / torch.sum(self.input_signals > 0, dim=-1)

    def approximate_entropy(self):
        apen = []
        # time_series_data = self.input_signals[0]
        m = 2
        r_list = [0.2 * torch.std(sig[sig > 0]) for sig in self.input_signals]
        # r_list = 0.2 * torch.std(self.input_signals, dim=-1)
        N_list = torch.sum(self.input_signals > 0, dim=-1)

        for sig, r, N in zip(self.input_signals, r_list, N_list):
            sig = sig[sig > 0]
            phi = np.zeros(N - m + 1)
            for i in tqdm(range(N - m + 1)):
                match_counts = 0
                for j in range(N - m + 1):
                    if i == j:
                        continue

                    dist = np.max(np.abs(sig[i:i + m] - sig[j:j + m]).numpy())
                    if dist <= r:
                        match_counts += 1

                phi[i] = match_counts / (N - m + 1)
            apen.append(-np.log(np.mean(phi)))
        return apen

    def report(self):
        plt.title(
            'Time Domain HRV Analysis \nMean HR(bpm): {:.2f} \nSDNN: {:.2f} \nRMSSD: {:.2f} \nNN50: {:.2f} \nPNN50: {:.2f} \nApproximate Entropy: {:.2f}'.format(
                self.hr, self.sdnn, self.rmssd, self.nn50, self.pnn50, self.apen))
        pass


class FrequencyHRVAnalysis:
    def __init__(self, input_signals: torch.tensor, fs=125.):
        self.input_signals = input_signals  # if torch.is_tensor(input_signals) else torch.tensor(input_signals)
        _, self.sig_len = self.input_signals.shape
        self.fs = fs
        self.amp, self.freq = self.fft()
        self.t_power = self.total_power()
        self.vlf_signal, self.vlf_power = self.vlf()
        self.lf_signal, self.lf_power = self.lf()
        self.hf_signal, self.hf_power = self.hf()
        self.rest_signal, self.rest_power = self.rest_f()
        self.norm_lf = self.normalized_lf()
        self.norm_hf = self.normalized_hf()
        self.lf_hf = self.lf_hf_ratio()

    def fft(self):
        fft_result = torch.fft.rfft(self.input_signals, dim=-1)
        frequencies = torch.fft.rfftfreq(self.sig_len, 1 / self.fs)
        return fft_result, frequencies

    def band_pass_filter(self, signal, f_low, f_high):
        fft_result = torch.fft.rfft(signal, dim=-1)
        frequencies = torch.fft.rfftfreq(self.sig_len, 1 / self.fs)
        bandpass_filter = torch.logical_and(frequencies >= f_low, frequencies <= f_high)
        filtered_fft = fft_result * bandpass_filter
        filtered_signal = torch.fft.irfft(filtered_fft)
        return torch.real(filtered_signal), torch.sum((torch.abs(filtered_fft) / (self.sig_len / 2)) ** 2, dim=-1)

    def total_power(self):
        return torch.sum((torch.abs(torch.fft.rfft(self.input_signals)) / (self.sig_len / 2)) ** 2, dim=-1)

    def vlf(self):  # 0.003 - 0.04 Hz
        return self.band_pass_filter(self.input_signals, 0.001, 0.04)

    def lf(self):  # 0.04 - 0.15 Hz
        return self.band_pass_filter(self.input_signals, 0.04, 0.15)

    def hf(self):  # 0.15 - 0.4 Hz
        return self.band_pass_filter(self.input_signals, 0.15, 0.4)

    def rest_f(self):
        return self.band_pass_filter(self.input_signals, 0.4, 2000)

    def lf_hf_ratio(self):
        return self.lf_power / self.hf_power

    def normalized_lf(self):
        return self.lf_power / (self.lf_power + self.hf_power)

    def normalized_hf(self):
        return self.hf_power / (self.lf_power + self.hf_power)

    def report(self):
        pass


def return_mean(target, prediction):
    return np.mean(np.vstack((target, prediction)), axis=0)


def return_diff(target, prediction):
    return np.array(target) - np.array(prediction)


class GraphicalReport:
    def __init__(self, targets: np.ndarray, preds: np.ndarray):
        self.preds = preds
        self.targets = targets
        self.color = ['royalblue', 'mediumseagreen', 'darkorange', 'firebrick', 'darkviolet']
        self.timehrv_target = TimeHRVAnalysis(input_signals=torch.tensor(self.targets))
        self.timehrv_pred = TimeHRVAnalysis(input_signals=torch.tensor(self.preds))
        self.freqhrv = FrequencyHRVAnalysis(input_signals=torch.tensor(self.targets))
        self.freqhrv_pred = FrequencyHRVAnalysis(input_signals=torch.tensor(self.preds))
        self.target_hr = self.timehrv_target.hr
        self.test_hr = self.timehrv_pred.hr

    def bland_altman_plot(self, models: list = None, title=None, xlabel=None, ylabel=None, show_plot=True):
        mean = return_mean(self.target_hr, self.test_hr)
        difference = return_diff(self.target_hr, self.test_hr)
        bias = np.mean(difference)
        std = np.std(difference)
        lower_limit = bias - 1.96 * std
        upper_limit = bias + 1.96 * std
        plt.figure(figsize=(10, 8))
        plt.title(title, fontsize=15, fontweight='bold')
        # for i in range(len(self.test)):
        plt.scatter(x=mean, y=difference, color=self.color[0], edgecolors='black', label=models[0])

        plt.axhline(y=bias, color='black', linestyle='--')
        plt.text((np.max(mean) - 12), (bias - 5), 'Mean: {}'.format(str(round(bias, 3))), color='black')
        plt.axhline(y=lower_limit, color='gray', linestyle='--')
        plt.text((np.max(mean) - 12), (lower_limit - 5), '-1.96SD: {}'.format(str(round(lower_limit, 3))),
                 color='black')
        plt.axhline(y=upper_limit, color='gray', linestyle='--')
        plt.text((np.max(mean) - 12), (upper_limit - 5), '+1.96SD: {}'.format(str(round(upper_limit, 3))),
                 color='black')
        plt.xlabel(xlabel, fontsize=10, fontweight='bold')
        plt.ylabel(ylabel, fontsize=10, fontweight='bold')
        plt.tight_layout()
        plt.legend(title='Model', loc='upper right', fontsize='small')
        plt.show()


def hrv_report(time, index_list, freq, info):
    for i in range(len(info)):
        hr = time.hr[i]
        sdnn = time.sdnn[i]
        rmssd = time.rmssd[i]
        nn50 = time.nn50[i]
        pnn50 = time.pnn50[i]
        # apen = time.apen[i]
        srd = time.srd[i]
        t_power = freq.t_power[i]
        vlf = freq.vlf_power[i]
        lf = freq.lf_power[i]
        hf = freq.hf_power[i]
        normalized_lf = freq.norm_lf[i]
        normalized_hf = freq.norm_hf[i]
        lf_hf_ratio = freq.lf_hf[i]
        fig, ax = plt.subplots(2, 2, gridspec_kw={'height_ratios': [1, 1], 'width_ratios': [2, 1]}, figsize=(10, 6))
        fig.suptitle('MIMIC-III HRV Report \n\nSubject ID: {} Hospital Admin ID: {} Age: {} Gender: {} Diagnosis: {}'.format(
            info[i][0], info[i][1], info[i][2], info[i][3], info[i][5]), fontsize=11, fontweight='bold')
        ax[0, 0].set_title('HRV Tachogram', fontsize=9, fontweight='bold')
        ax[0, 0].set_xlabel('Time (min)')
        ax[0, 0].set_ylabel('HR (bpm)')
        ax[0, 0].set_ylim(40, 140)
        ax[0, 0].text(0, 130,
                      'mean RRI: ' + str(np.round(np.mean(time.input_signals[i][time.input_signals[i] > 0].numpy()),
                                                  1)) + '(ms), mean HR:' + str(
                          np.round(np.mean(hr.numpy()), 1)) + '(bpm)')
        # ax[0, 0].plot(time.input_signals[i][time.input_signals[i] > 0], color='royalblue',
        #               label='Input Signal')
        t = (index_list[i][index_list[i] > 0] / 18000) * 5
        t = t[:len(time.input_signals[i][time.input_signals[i] > 0])]
        hr_tacho = 60 / (time.input_signals[i][time.input_signals[i] > 0] / 1000)
        max_hr = np.max(hr_tacho.numpy())
        min_hr = np.min(hr_tacho.numpy())
        ax[0, 0].fill_between(t, np.ones(len(t)) * 60, np.ones(len(t)) * 100, color='lightgray', alpha=0.5)
        ax[0, 0].plot(t, hr_tacho, color='royalblue')
        ax[0, 0].axhline(y=max_hr, xmin=0.05, xmax=0.95, color='orange', linestyle='--', linewidth=0.9, label='Max HR')
        ax[0, 0].axhline(y=min_hr, xmin=0.05, xmax=0.95, color='orange', linestyle='-.', linewidth=0.9, label='Min HR')
        # ax[0, 0].text(0, max_hr + 2, str(np.round(max_hr, 2)))
        # ax[0, 0].text(0, min_hr - 7, str(np.round(min_hr, 2)))
        ax[0, 0].legend(loc='upper right', fontsize='small')
        bar_plot_x = np.arange(2)
        bar_plot_x_values = [sdnn, rmssd]
        ax[0, 1].set_title('Time Domain Components', fontsize=9, fontweight='bold')
        ax[0, 1].bar(bar_plot_x, bar_plot_x_values,
                     color=['mediumseagreen', 'darkorange'], width=0.4)
        ax[0, 1].set_xticks(bar_plot_x, ['SDNN', 'RMSSD'])
        ax[0, 1].hlines(y=30, xmin=-0.1, xmax=0.1, color='gray')
        ax[0, 1].hlines(y=50, xmin=-0.1, xmax=0.1, color='gray')
        ax[0, 1].hlines(y=20, xmin=0.9, xmax=1.1, color='gray')
        ax[0, 1].hlines(y=40, xmin=0.9, xmax=1.1, color='gray')
        ax[0, 1].vlines(x=0, ymin=30, ymax=50, color='gray')
        ax[0, 1].vlines(x=1, ymin=20, ymax=40, color='gray')
        ax[0, 1].hlines(y=20, xmin=-0.1, xmax=0.1, color='red')
        ax[0, 1].hlines(y=10, xmin=0.9, xmax=1.1, color='red')
        ax[0, 1].text(0, sdnn, str(np.round(sdnn.numpy(), 3)), ha='center', va='bottom')
        ax[0, 1].text(1, rmssd, str(np.round(rmssd.numpy(), 3)), ha='center', va='bottom')
        # ax[0, 1].axes.yaxis.set_visible(False)
        ax[1, 0].set_title('Signal Decomposition', fontsize=9, fontweight='bold')
        sig_t = np.arange(0, 5, 5 / 18000)
        ax[1, 0].plot(sig_t, freq.hf_signal[i], color='darkorange', label='HF: 0.15~0.4 Hz')
        ax[1, 0].plot(sig_t, freq.lf_signal[i], color='mediumseagreen', label='LF: 0.04~0.15 Hz')
        ax[1, 0].plot(sig_t, freq.vlf_signal[i], color='royalblue', label='VLF: 0.003~0.04 Hz')
        ax[1, 0].set_xlabel('Time (min)')
        ax[1, 0].legend(loc='upper right', fontsize='small')
        ax[1, 0].axes.yaxis.set_visible(False)

        ax[1, 1].set_title('Frequency Domain Components', fontsize=9, fontweight='bold')
        bar_plot_x = np.arange(3)
        bar_plot_x_values = [vlf, lf, hf]
        ax[1, 1].bar(bar_plot_x, bar_plot_x_values,
                     color=['royalblue', 'mediumseagreen', 'darkorange'], width=0.4)
        ax[1, 1].set_xticks(bar_plot_x, ['VLF', 'LF', 'HF'])
        ax[1, 1].set_ylim(0, np.max(bar_plot_x_values) * 1.3)
        ax[1, 1].text(1, lf, 'LF Norm\n' + str(np.round(normalized_lf.numpy(), 3)), ha='center', va='bottom')
        ax[1, 1].text(2, hf, 'HF Norm\n' + str(np.round(normalized_hf.numpy(), 3)), ha='center', va='bottom')
        ax[1, 1].text(1.5, (lf + hf) / 2, 'LF/HF\n' + str(np.round(lf_hf_ratio.numpy(), 2)), ha='center', va='bottom')
        ax[1, 1].axes.yaxis.set_visible(False)
        fig.tight_layout()
        # plt.show()
        plt.savefig('./hrv_img/'+str('Subject ID: {} Hospital Admin ID: {} Age: {} Gender: {} Diagnosis: {}'.format(
            info[i][0], info[i][1], info[i][2], info[i][3], info[i][5]))+'.png')
        plt.close()


if __name__ == "__main__":
    FS = 60.
    pleth = pd.read_csv('/home/paperc/PycharmProjects/rppg/cnibp/preprocessing/pleth.csv', header=None).to_numpy()[1:,
            1:]
    info = pd.read_csv('/home/paperc/PycharmProjects/rppg/cnibp/preprocessing/info.csv').to_numpy()[:, 1:]

    # pleth = np.unique(pleth, axis=0)
    # info = np.unique(info, axis=0)

    # pleth = pleth[:-1]
    # info = info[:-1]

    hr_list, hrv_list, index_list = calc_hr_torch('PEAK', detrend_torch(torch.tensor(pleth)), FS)
    freq = FrequencyHRVAnalysis(input_signals=torch.tensor(pleth), fs=FS)
    time = TimeHRVAnalysis(input_signals=torch.tensor(hrv_list), fs=FS)
    # cnt = 0
    # fig, ax = plt.subplots(3, 1, gridspec_kw={'height_ratios': [1, 1, 1]}, figsize=(10, 6))
    # plt.show()
    hrv_report(time, index_list, freq, info)
