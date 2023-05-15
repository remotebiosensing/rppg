import numpy as np
import scipy.signal
from matplotlib import pyplot as plt
from scipy.signal import butter
from scipy.sparse import spdiags
import torch
def detrend(signal, Lambda):
    """detrend(signal, Lambda) -> filtered_signal
    This function applies a detrending filter.
    This  is based on the following article "An advanced detrending method with application
    to HRV analysis". Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.
    *Parameters*
      ``signal`` (1d numpy array):
        The signal where you want to remove the trend.
      ``Lambda`` (int):
        The smoothing parameter.
    *Returns*
      ``filtered_signal`` (1d numpy array):
        The detrended signal.
    """
    signal_length = len(signal)

    # observation matrix
    H = np.identity(signal_length)

    # second-order difference matrix

    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot((H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), signal)
    return filtered_signal


def BPF(input_val, fs=30,low= 0.75, high=2.5):
    low = low / (0.5 * fs)
    high = high / (0.5 * fs)
    [b_pulse, a_pulse] = butter(1, [low, high], btype='bandpass')
    return scipy.signal.filtfilt(b_pulse, a_pulse, np.double(input_val))


def plot_graph(start_point, length, target, inference):
    plt.rcParams["figure.figsize"] = (14, 5)
    plt.plot(range(len(target[start_point:start_point + length])), target[start_point:start_point + length],
             label='target')
    plt.plot(range(len(inference[start_point:start_point + length])), inference[start_point:start_point + length],
             label='inference')
    plt.legend(fontsize='x-large')
    plt.show()


def normalize(input_val):
    return (input_val - np.mean(input_val)) / np.std(input_val)

def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def calculate_hr(cal_type, ppg_signal, fs=60, low_pass=0.75, high_pass=2.5):
    """Calculate heart rate based on PPG using Fast Fourier transform (FFT)."""
    if cal_type == "FFT":
        ppg_signal = np.expand_dims(ppg_signal, 0)
        N = _next_power_of_2(ppg_signal.shape[1])
        f_ppg, pxx_ppg = scipy.signal.periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
        fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
        mask_ppg = np.take(f_ppg, fmask_ppg)
        mask_pxx = np.take(pxx_ppg, fmask_ppg)
        hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
    else:
        ppg_peaks, _ = scipy.signal.find_peaks(ppg_signal)
        hr = 60 / (np.mean(np.diff(ppg_peaks)) / fs)
    return hr

def mag2db(magnitude):
    return 20. * np.log10(magnitude)

def get_hr(pred, label, model_type, cal_type, fs=30, bpf_flag=True,low =0.75,high=2.5):
    if model_type == "DIFF":
        pred = detrend(np.cumsum(pred),100)
        label = detrend(np.cumsum(label),100)
    else:
        pred = detrend(pred,100)
        label = detrend(label,100)

    if bpf_flag:
        pred = BPF(pred,fs,low,high)
        label = BPF(pred,fs,low,high)

    if cal_type != "BOTH":
        hr_pred = [calculate_hr(cal_type,p,fs,low,high) for p in pred]
        hr_label = [calculate_hr(cal_type,l,fs,low,high) for l in label]
    else:
        hr_pred_fft = calculate_hr("FFT", pred, fs, low, high)
        hr_label_fft = calculate_hr("FFT", label, fs, low, high)
        hr_pred_peak = calculate_hr("PEAK", pred, fs, low, high)
        hr_label_peak = calculate_hr("PEAK", label, fs, low, high)
        hr_pred = [hr_pred_fft,hr_pred_peak]
        hr_label = [hr_label_fft, hr_label_peak]

    return hr_pred, hr_label

def MAE(pred,label):
    return np.mean(np.abs(pred-label))

def RMSE(pred,label):
    return np.sqrt(np.mean(np.square(pred-label)))

def MAPE(pred,label):
    return np.mean(np.abs((pred-label)/label)) *100

def corr(pred,label):
    return np.corrcoef(pred,label)




class IrrelevantPowerRatio(torch.nn.Module):
    # we reuse the code in Gideon2021 to get irrelevant power ratio
    # Gideon, John, and Simon Stent. "The way to my heart is through contrastive learning: Remote photoplethysmography from unlabelled video." Proceedings of the IEEE/CVF international conference on computer vision. 2021.
    def __init__(self, Fs, high_pass, low_pass):
        super(IrrelevantPowerRatio, self).__init__()
        self.Fs = Fs
        self.high_pass = high_pass
        self.low_pass = low_pass

    def forward(self, preds):
        # Get PSD
        X_real = torch.view_as_real(torch.fft.rfft(preds, dim=-1, norm='forward'))

        # Determine ratio of energy between relevant and non-relevant regions
        Fn = self.Fs / 2
        freqs = torch.linspace(0, Fn, X_real.shape[-2])
        use_freqs = torch.logical_and(freqs >= self.high_pass / 60, freqs <= self.low_pass / 60)
        zero_freqs = torch.logical_not(use_freqs)
        use_energy = torch.sum(torch.linalg.norm(X_real[:,use_freqs], dim=-1), dim=-1)
        zero_energy = torch.sum(torch.linalg.norm(X_real[:,zero_freqs], dim=-1), dim=-1)
        denom = use_energy + zero_energy
        energy_ratio = torch.ones_like(denom)
        for ii in range(len(denom)):
            if denom[ii] > 0:
                energy_ratio[ii] = zero_energy[ii] / denom[ii]
        return energy_ratio