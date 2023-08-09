import torch
import numpy as np
from scipy import signal
from scipy.sparse import spdiags

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


def detrend_torch(signals, Lambda=100):
    """
    Detrend 1D signals with diagonal matrix D, using torch batch matrix multiplication

    :param signals: Singals with linear trend
    :param Lambda:
    :return:
    """
    test_n, length = signals.shape
    signals = signals.to(torch.float).to('cuda')

    H = torch.eye(length).to('cuda')
    ones = torch.diag(torch.ones(length - 2)).to('cuda')
    zeros_1 = torch.zeros((length - 2, 1)).to('cuda')
    zeros_2 = torch.zeros((length - 2, 2)).to('cuda')

    D = torch.cat((ones, zeros_2), dim=-1) +\
        torch.cat((zeros_1, -2 * ones, zeros_1), dim=-1) +\
        torch.cat((zeros_2, ones), dim=-1)

    detrended_signal = torch.bmm(signals.unsqueeze(1),
                                 (H - torch.linalg.inv(H + (Lambda ** 2) * torch.t(D) @ D)).expand(test_n, -1, -1)).squeeze()
    detrended_signal += torch.mean(signals, dim=-1, keepdim=True)
    return detrended_signal


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

def watch_peaks(raw_signals, index_list, num, start_idx, end_idx):
    raw_signals = raw_signals[num].cpu().numpy()
    index_list = index_list[num].to(torch.int).cpu().numpy()
    index_list = index_list[np.logical_and(index_list > start_idx, index_list < end_idx)]
    index_list = peak_correction(index_list)
    plt.plot(np.arange(start_idx, end_idx, 1), raw_signals[start_idx:end_idx])
    plt.plot(index_list, raw_signals[index_list], 'ro')
    plt.show()


def peak_correction(abnormal_indices, device='cuda'):
    std_list = []
    if device == 'cpu':
        diff_mean = np.mean(np.diff(abnormal_indices))
        bad_idx = np.where(np.diff(abnormal_indices) < 0.8 * diff_mean)[0] + 1
        if len(bad_idx) == 0:
            return abnormal_indices
        else:
            for i in bad_idx:
                std_list.append(np.std(np.diff(np.delete(abnormal_indices, i))))
            corrected_indices = np.delete(abnormal_indices, bad_idx[np.argmin(std_list)])
            return corrected_indices
    else:
        diff_mean = torch.mean(torch.diff(abnormal_indices).to(torch.float))
        bad_idx = torch.split(torch.where(torch.diff(abnormal_indices) < 0.8 * diff_mean)[0] + 1, 2)
        remove_cnt = 0
        if len(bad_idx[0]) == 0:
            return abnormal_indices
        else:
            for pair in bad_idx:
                if len(pair) == 1:
                    abnormal_indices = torch.cat((abnormal_indices[:pair[0]], abnormal_indices[pair[0]+1:]))
                    return abnormal_indices
                else:
                    pair -= remove_cnt
                    roi = abnormal_indices[pair[0]-5:pair[1]+5]
                    roi_pair = [5, 6]
                    std_1 = torch.std(torch.diff(torch.cat((roi[:roi_pair[0]], roi[roi_pair[0]+1:]))).to(torch.float))
                    std_2 = torch.std(torch.diff(torch.cat((roi[:roi_pair[1]], roi[roi_pair[1]+1:]))).to(torch.float))
                    if std_1 < std_2:
                        abnormal_indices = torch.cat((abnormal_indices[:pair[0]], abnormal_indices[pair[0]+1:]))
                    else:
                        abnormal_indices = torch.cat((abnormal_indices[:pair[1]], abnormal_indices[pair[1]+1:]))
                    remove_cnt += 1
                # if std_1 < std_2:

                # for i in pair:
                #     std_list.append(torch.std(torch.diff(torch.cat(abnormal_indices[:i], abnormal_indices[i+1:]))))
                # corrected_indices = torch.cat(abnormal_indices[:bad_idx[torch.argmin(torch.tensor(std_list))]], abnormal_indices[bad_idx[torch.argmin(torch.tensor(std_list))]+1:])
            return abnormal_indices


def calc_hr_torch(calc_type, ppg_signals, fs=30., report_flag=False):
    if type(ppg_signals) != torch.Tensor:
        ppg_signals = torch.from_numpy(ppg_signals.copy())
    if torch.cuda.is_available():
        ppg_signals = ppg_signals.to('cuda')
    test_n, sig_length = ppg_signals.shape
    hr_list = torch.empty(test_n)
    if calc_type == "FFT":
        ppg_signals = ppg_signals - torch.mean(ppg_signals, dim=-1, keepdim=True)
        N = sig_length
        k = torch.arange(N)
        T = N / fs
        freq = k / T
        amplitude = torch.abs(torch.fft.rfft(ppg_signals, n=N, dim=-1)) / N

        hr_list = freq[torch.argmax(amplitude, dim=-1)] * 60

        return hr_list
    else:  # calc_type == "Peak"
        hrv_list = -torch.ones((test_n, int(sig_length // fs) * 10)).to('cuda')
        index_list = -torch.ones((test_n, int(sig_length // fs) * 10)).to('cuda')
        width = 11  # odd / physnet(5), diff (11)
        window_maxima = torch.nn.functional.max_pool1d(ppg_signals, width, 1, padding=width // 2, return_indices=True)[
            1].to('cuda')
        window_minima = torch.nn.functional.max_pool1d(-ppg_signals, width, 1, padding=width // 2, return_indices=True)[
            1].to('cuda')
        # if window_maxima.dim() == 1:
        #     window_maxima = window_maxima.unsqueeze(0)
        for i in range(test_n):
            peak_candidate = window_maxima[i].unique()
            nice_peak = peak_candidate[window_maxima[i][peak_candidate] == peak_candidate]
            valley_candidate = window_minima[i].unique()
            nice_valley = valley_candidate[window_minima[i][valley_candidate] == valley_candidate]
            # thresholding
            nice_peak = nice_peak[
                ppg_signals[i][nice_peak] > torch.mean(ppg_signals[i][nice_peak]) * 0.8]  # peak thresholding
            nice_valley = nice_valley[
                ppg_signals[i][nice_valley] < torch.mean(ppg_signals[i][nice_valley]) * 1.2]  # min thresholding
            if len(nice_peak) / len(nice_valley) > 1.8:  # remove false peaks
                if torch.all(nice_peak[:2] > nice_valley[0]):
                    nice_peak = nice_peak[0::2]
                else:
                    nice_peak = nice_peak[1::2]
            nice_peak = peak_correction(nice_peak, 'cuda')
            beat_interval = torch.diff(nice_peak)  # sample
            hrv = beat_interval / fs  # second
            hr = torch.mean(60 / hrv)
            hr_list[i] = hr
            hrv_list[i, :len(hrv)] = hrv * 1000  # milli second
            index_list[i, :len(nice_peak)] = nice_peak


        hrv_list = hrv_list[:, :torch.max(torch.sum(hrv_list > 0, dim=-1))]
        index_list = index_list[:, :torch.max(torch.sum(index_list > 0, dim=-1))]
        # watch_peaks(ppg_signals, index_list, 1, 12000, 14000)
        return hr_list.cpu(), hrv_list.cpu(), index_list.to(torch.long).cpu()


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


def get_hr(rppg, bvp, model_type, vital_type='HR', cal_type='FFT', fs=30, bpf=None, report_flag=False):
    if cal_type == 'FFT' and vital_type == 'HRV':
        raise ValueError("'FFT' cannot calculate HRV. To calculate HRV, use 'PEAK' method instead.")
    if cal_type not in ['FFT', 'PEAK']:
        raise NotImplementedError("cal_type must be 'FFT' or 'PEAK'.")

    if model_type == "DIFF":
        bvp = detrend_torch(torch.cumsum(bvp, dim=-1))
        rppg = detrend_torch(torch.cumsum(rppg, dim=-1))
    else:
        bvp = detrend_torch(bvp)
        rppg = detrend_torch(rppg)

    if bpf != 'None':
        low, high = bpf
        bvp = normalize_torch(BPF(bvp, fs, low, high))
        rppg = normalize_torch(BPF(rppg, fs, low, high))
    else:
        bvp = normalize_torch(bvp)
        rppg = normalize_torch(rppg)

    # TODO: torch bpf
    hr_target = calc_hr_torch(cal_type, bvp, fs, report_flag)
    hr_pred = calc_hr_torch(cal_type, rppg, fs, report_flag)

    if cal_type == 'PEAK':
        hr_target, hrv_target, index_target = hr_target
        hr_pred, hrv_pred, index_pred = hr_pred
        if vital_type == 'HRV':
            return hrv_pred, hrv_target

    return hr_pred, hr_target


def MAE(pred, label):
    return np.mean(np.abs(pred - label))


def RMSE(pred, label):
    return np.sqrt(np.mean((pred - label) ** 2))


def MAPE(pred, label):
    return np.mean(np.abs((pred - label) / label)) * 100


def corr(pred, label):
    return np.corrcoef(pred, label)


def SD(pred, label):
    return np.std(pred - label)


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
