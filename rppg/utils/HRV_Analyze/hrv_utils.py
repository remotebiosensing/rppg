import numpy as np
import torch
import matplotlib.pyplot as plt


class TimeHRVAnalysis:
    def __init__(self, input_signals: torch.tensor, fs: float = 30.):
        self.mask = torch.tensor(input_signals > 0, dtype=torch.int)
        self.input_signals = self.remove_negative(input_signals)
        self.fs = fs

        self.hr = self.mean_hr()
        self.sdnn = self.calculate_sdnn()
        self.rmssd = self.calculate_rmssd()
        self.nn50 = self.calculate_nn50()
        self.pnn50 = self.calculate_pnn50()

    def remove_negative(self, input_signals):
        max_len = torch.max(torch.sum(input_signals > 0, dim=-1))
        # arg_max = torch.argmax(torch.sum(input_signals > 0, dim=-1))
        return input_signals[:, :max_len]

    def mean_hr(self):
        hr = []
        for hrv in self.input_signals:
            hr.append(torch.mean(60 / (hrv[hrv > 0] / 1000)))

        return hr

    def calculate_sdnn(self):
        # temp = self.input_signals
        sdnn = []
        for hrv in self.input_signals:
            sdnn.append(torch.std(hrv[hrv > 0]))

        return sdnn

    def calculate_rmssd(self):
        rmssd = []
        for hrv in self.input_signals:
            diff = torch.diff(hrv[hrv > 0])
            rmssd.append(torch.sqrt(torch.mean(torch.square(diff))))
        return rmssd

    def calculate_nn50(self):
        nn50 = []
        for hrv in self.input_signals:
            diff = torch.abs(torch.diff(hrv[hrv > 0]))
            nn50.append(torch.sum(diff > 50))

        return nn50

    def calculate_pnn50(self):
        pnn50 = []
        for hrv in self.input_signals:
            diff = torch.abs(torch.diff(hrv[hrv > 0]))
            pnn50.append(torch.sum(diff > 50) / len(diff))

        return pnn50

    def report(self):
        pass


class FrequencyHRVAnalysis:
    def __init__(self, input_signals: torch.tensor, fs: float = 30.):
        self.input_signals = input_signals

    def total_power(self):
        pass

    def vlf(self):
        pass

    def lf(self):
        pass

    def hf(self):
        pass

    def lf_hf_ratio(self):
        pass

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
        plt.legend(title='Model', loc='upper right')
        plt.show()
        pass
