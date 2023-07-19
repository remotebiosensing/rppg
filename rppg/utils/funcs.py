import torch
import numpy as np
from scipy import signal
from scipy.sparse import spdiags
from rppg.utils.visualization import plot


from matplotlib import pyplot as plt
import neurokit2 as nk


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


def detrend_torch(signals, Lambda):
    test_n, length = signals.shape

    H = torch.eye(length)
    ones = torch.ones(length - 2)

    diag1 = torch.cat((torch.diag(ones), torch.zeros((length - 2, 2))), dim=-1)
    diag2 = torch.cat((torch.zeros((length - 2, 1)), torch.diag(-2 * ones), torch.zeros((length - 2, 1))), dim=-1)
    diag3 = torch.cat((torch.zeros((length - 2, 2)), torch.diag(ones)), dim=-1)
    D = diag1 + diag2 + diag3

    filtered_signal = torch.bmm(signals.unsqueeze(1),
                                (H - torch.linalg.inv(H + (Lambda ** 2) * torch.t(D) @ D)).to('cuda').expand(test_n, -1,
                                                                                                             -1)).squeeze()
    return filtered_signal


def BPF(input_val, fs=30, low=0.75, high=2.5):
    low = low / (0.5 * fs)
    high = high / (0.5 * fs)
    [b_pulse, a_pulse] = signal.butter(6, [low, high], btype='bandpass')
    if type(input_val) == torch.Tensor:
        return signal.filtfilt(b_pulse, a_pulse, np.double(input_val.cpu().numpy()))
    else:
        return signal.filtfilt(b_pulse, a_pulse, np.double(input_val))


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

def normalize_torch(input_val):
    if type(input_val) != torch.Tensor:
        input_val = torch.from_numpy(input_val.copy())
    min = torch.min(input_val, dim=-1, keepdim=True)[0]
    max = torch.max(input_val, dim=-1, keepdim=True)[0]
    return (input_val - min) / (max - min)


def _nearest_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def get_hrv(ppg_signal, fs=30.):
    ppg_peaks = nk.ppg_findpeaks(ppg_signal, sampling_rate=fs)['PPG_Peaks']
    hrv = nk.signal_rate(ppg_peaks, sampling_rate=fs, desired_length=len(ppg_signal))
    return hrv


def calc_hr_torch(calc_type, ppg_signals, fs=30., low_freq=0.75, high_freq=2.5):  # hr range 45 ~ 150
    # calc_type = 'Peak'
    test_n, sig_length = ppg_signals.shape
    hr_list = torch.empty(test_n)
    if calc_type == "FFT":
        N = _nearest_power_of_2(sig_length)
        psd = torch.abs(torch.fft.rfft(ppg_signals, n=N, dim=-1) ** 2)
        freq = torch.linspace(0, 15, len(psd[0])).cuda()
        f_mask = (freq >= low_freq) & (freq <= high_freq)
        freq = freq[f_mask]
        psd = psd[:, f_mask]
        hr_list = freq[torch.argmax(psd, dim=-1)] * 60

        return hr_list
    else:
        hrv_list = torch.zeros((test_n, sig_length // fs * 4))
        index_list = torch.zeros((test_n, sig_length // fs * 4))
        width = 11  # odd (11)
        window_maxima = torch.nn.functional.max_pool1d(ppg_signals, width, 1, padding=width // 2, return_indices=True)[
            1].squeeze()
        # candidates = [x.unique() for x in window_maxima]

        for i in range(test_n):
            candidate = window_maxima[i].unique()
            nice_peaks = candidate[window_maxima[i][candidate] == candidate]
            nice_peaks = nice_peaks[
                ppg_signals[i][nice_peaks] > torch.mean(ppg_signals[i][nice_peaks] / 2)]  # threshold
            beat_interval = torch.diff(nice_peaks)   # sample
            hrv = beat_interval / fs  # second
            hr = torch.mean(60 / hrv)
            hr_list[i] = hr
            hrv_list[i, :len(hrv)] = hrv * 1000  # milli second
            index_list[i, :len(nice_peaks)] = nice_peaks

        return hr_list, hrv_list, index_list


def calculate_hr(cal_type, ppg_signal, fs=60., low_pass=0.75, high_pass=2.5):
    """Calculate heart rate based on PPG using Fast Fourier transform (FFT)."""
    if cal_type == "FFT":
        ppg_signal = np.expand_dims(ppg_signal, 0)
        N = _nearest_power_of_2(ppg_signal.shape[1])
        f_ppg, pxx_ppg = signal.periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
        fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
        mask_ppg = np.take(f_ppg, fmask_ppg)
        mask_pxx = np.take(pxx_ppg, fmask_ppg)
        hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60

    else:
        hrv = get_hrv(ppg_signal, fs=fs)
        hr = np.mean(hrv, dtype=np.float32)
        # ppg_peaks, _ = scipy.signal.find_peaks(ppg_signal)
        # hr = 60 / (np.mean(np.diff(ppg_peaks)) / fs)
    return hr


def mag2db(magnitude):
    return 20. * np.log10(magnitude)


def get_hr(pred, target, model_type, cal_type='FFT', fs=30, bpf_flag=True, low=0.75, high=2.5):
    if model_type == "DIFF":
        target = detrend_torch(torch.cumsum(target, dim=-1), 100)
        pred = detrend_torch(torch.cumsum(pred, dim=-1), 100)
    else:
        target = detrend_torch(target, 100)
        pred = detrend_torch(pred, 100)
    if bpf_flag:
        f_target = BPF(target, fs, low, high)
        f_pred = BPF(pred, fs, low, high)

    # TODO: torch bpf
    if cal_type == 'FFT':
        hr_target = calc_hr_torch('FFT', target, fs, low, high)
        hr_pred = calc_hr_torch('FFT', pred, fs, low, high)
        if bpf_flag:
            f_hr_target = calc_hr_torch('FFT', torch.from_numpy(f_target.copy()), fs, low, high)
            f_hr_pred = calc_hr_torch('FFT', torch.from_numpy(f_pred.copy()), fs, low, high)
        return hr_pred, hr_target
    elif cal_type == 'PEAK':
        hr_target, hrv_target, index_target = calc_hr_torch('PEAK', target, fs, low, high)
        hr_pred, hrv_pred, index_pred = calc_hr_torch('PEAK', pred, fs, low, high)
        if bpf_flag:
            f_hr_target, f_hrv_target, f_index_target = calc_hr_torch('PEAK', torch.from_numpy(f_target.copy()), fs, low, high)
            f_hr_pred, f_hrv_pred, f_index_pred = calc_hr_torch('PEAK', torch.from_numpy(f_pred.copy()), fs, low, high)
            pred = normalize_torch(pred)
            f_pred = normalize_torch(f_pred)
            target = normalize_torch(target)
            f_target = normalize_torch(f_target)
            for i in range(len(pred)):
                plot(pred[i], hr_pred[i], hrv_pred[i], index_pred[i], f_pred[i], f_hr_pred[i], f_hrv_pred[i], f_index_pred[i],
                     target[i], hr_target[i], hrv_target[i], index_target[i], f_target[i], f_hr_target[i], f_hrv_target[i], f_index_target[i])
        return [hr_pred, hrv_pred], [hr_target, hrv_target]
    else:
        hr_pred_fft = calc_hr_torch('FFT', pred, fs, low, high)
        hr_label_fft = calc_hr_torch('FFT', target, fs, low, high)
        hr_pred_peak = calc_hr_torch('PEAK', pred, fs, low, high)
        hr_label_peak = calc_hr_torch('PEAK', target, fs, low, high)
        hr_pred = [hr_pred_fft, hr_pred_peak]
        hr_target = [hr_label_fft, hr_label_peak]


def MAE(pred, label):
    return np.mean(np.abs(pred - label))


def RMSE(pred, label):
    return np.sqrt(np.mean((pred - label) ** 2))


def MAPE(pred, label):
    return np.mean(np.abs((pred - label) / label)) * 100


def corr(pred, label):
    return np.corrcoef(pred, label)


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
        use_energy = torch.sum(torch.linalg.norm(X_real[:, use_freqs], dim=-1), dim=-1)
        zero_energy = torch.sum(torch.linalg.norm(X_real[:, zero_freqs], dim=-1), dim=-1)
        denom = use_energy + zero_energy
        energy_ratio = torch.ones_like(denom)
        for ii in range(len(denom)):
            if denom[ii] > 0:
                energy_ratio[ii] = zero_energy[ii] / denom[ii]
        return energy_ratio


def sinc_impulse_response(cutoff: torch.Tensor, window_size: int = 513, high_pass: bool = False):
    # https://github.com/pytorch/audio/blob/main/torchaudio/prototype/functional/_dsp.py
    """Create windowed-sinc impulse response for given cutoff frequencies.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        cutoff (Tensor): Cutoff frequencies for low-pass sinc filter.

        window_size (int, optional): Size of the Hamming window to apply. Must be odd.
        (Default: 513)

        high_pass (bool, optional):
            If ``True``, convert the resulting filter to high-pass.
            Otherwise low-pass filter is returned. Default: ``False``.

    Returns:
        Tensor: A series of impulse responses. Shape: `(..., window_size)`.
    """
    if window_size % 2 == 0:
        raise ValueError(f"`window_size` must be odd. Given: {window_size}")

    half = window_size // 2
    device, dtype = cutoff.device, cutoff.dtype
    idx = torch.linspace(-half, half, window_size, device=device, dtype=dtype)

    filt = torch.special.sinc(cutoff.unsqueeze(-1) * idx.unsqueeze(0))
    filt = filt * torch.hamming_window(window_size, device=device, dtype=dtype, periodic=False).unsqueeze(0)
    filt = filt / filt.sum(dim=-1, keepdim=True).abs()

    # High pass IR is obtained by subtracting low_pass IR from delta function.
    # https://courses.engr.illinois.edu/ece401/fa2020/slides/lec10.pdf
    if high_pass:
        filt = -filt
        filt[..., half] = 1.0 + filt[..., half]
    return filt
