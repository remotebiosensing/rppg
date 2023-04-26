import os

import torch
import torch.nn as nn
import torch.nn.modules.loss as loss
import numpy as np
from log import log_warning
import torch.nn.functional as F
import scipy
from numpy import dot
from numpy.linalg import norm

import torch.utils.checkpoint as cp
from params import params
from scipy.signal import find_peaks
from scipy.signal import butter, sosfiltfilt
from rppg.utils.funcs import _next_power_of_2
def loss_fn():
    """
    :param loss_fn: implement loss function for training
    :return: loss function module(class)
    """
    if params.log_flag:
        print("========= loss_fn() in" + os.path.basename(__file__))

    if params.loss_fn == "mse":
        return loss.MSELoss()
    elif params.loss_fn == "fft":
        return fftLoss()
    elif params.loss_fn == "L1":
        return loss.L1Loss()
    elif params.loss_fn == "neg_pearson":
        return NegPearsonLoss()
    elif params.loss_fn == "multi_margin":
        return loss.MultiMarginLoss()
    elif params.loss_fn == "bce":
        return loss.BCELoss()
    elif params.loss_fn == "huber":
        return loss.HuberLoss()
    elif params.loss_fn == "cosine_embedding":
        return loss.CosineEmbeddingLoss()
    elif params.loss_fn == "cross_entropy":
        return loss.CrossEntropyLoss()
    elif params.loss_fn == "ctc":
        return loss.CTCLoss()
    elif params.loss_fn == "bce_with_logits":
        return loss.BCEWithLogitsLoss()
    elif params.loss_fn == "gaussian_nll":
        return loss.GaussianNLLLoss()
    elif params.loss_fn == "hinge_embedding":
        return loss.HingeEmbeddingLoss()
    elif params.loss_fn == "KLDiv":
        return loss.KLDivLoss()
    elif params.loss_fn == "margin_ranking":
        return loss.MarginRankingLoss()
    elif params.loss_fn == "multi_label_margin":
        return loss.MultiLabelMarginLoss()
    elif params.loss_fn == "multi_label_soft_margin":
        return loss.MultiLabelSoftMarginLoss()
    elif params.loss_fn == "nll":
        return loss.NLLLoss()
    elif params.loss_fn == "nll2d":
        return loss.NLLLoss2d()
    elif params.loss_fn == "pairwise":
        return loss.PairwiseDistance()
    elif params.loss_fn == "poisson_nll":
        return loss.PoissonNLLLoss()
    elif params.loss_fn == "smooth_l1":
        return loss.SmoothL1Loss()
    elif params.loss_fn == "soft_margin":
        return loss.SoftMarginLoss()
    elif params.loss_fn == "triplet_margin":
        return loss.TripletMarginLoss()
    elif params.loss_fn == "triplet_margin_distance":
        return loss.TripletMarginWithDistanceLoss()
    elif params.loss_fn == "RhythmNetLoss":
        return RhythmNetLoss()
    elif params.loss_fn == "stftloss":
        return stftLoss()
    elif params.loss_fn == "pearson":
        return PearsonLoss()
    elif params.loss_fn == "BVPVelocityLoss":
        return BVPVelocityLoss()
    else:
        log_warning("use implemented loss functions")
        raise NotImplementedError("implement a custom function(%s) in loss.py" % loss_fn)


def neg_Pearson_Loss(predictions, targets):
    '''
    :param predictions: inference value of trained model
    :param targets: target label of input data
    :return: negative pearson loss
    '''
    rst = 0
    targets = targets[:, :]
    predictions = torch.squeeze(predictions)
    # Pearson correlation can be performed on the premise of normalization of input data
    predictions = (predictions - torch.mean(predictions)) / torch.std(predictions)
    targets = (targets - torch.mean(targets)) / torch.std(targets)

    for i in range(predictions.shape[0]):
        sum_x = torch.sum(predictions[i])  # x
        sum_y = torch.sum(targets[i])  # y
        sum_xy = torch.sum(predictions[i] * targets[i])  # xy
        sum_x2 = torch.sum(torch.pow(predictions[i], 2))  # x^2
        sum_y2 = torch.sum(torch.pow(targets[i], 2))  # y^2
        N = predictions.shape[1] if len(predictions.shape) > 1 else 1
        pearson = (N * sum_xy - sum_x * sum_y) / (
            torch.sqrt((N * sum_x2 - torch.pow(sum_x, 2)) * (N * sum_y2 - torch.pow(sum_y, 2))))

        rst += 1 - pearson

    rst = rst / predictions.shape[0]
    return rst


def peak_mse(predictions, targets):
    rst = 0
    targets = targets[:, :]


class NegPearsonLoss(nn.Module):
    def __init__(self):
        super(NegPearsonLoss, self).__init__()

    def forward(self, predictions, targets):
        return neg_Pearson_Loss(predictions, targets)


class fftLoss(nn.Module):
    def __init__(self):
        super(fftLoss, self).__init__()

    def forward(self, predictions, targets):
        neg = neg_Pearson_Loss(predictions, targets)
        loss_func = nn.L1Loss()
        predictions = torch.fft.fft(predictions, dim=1, norm="forward")
        targets = torch.fft.fft(targets, dim=1, norm="forward")
        loss = loss_func(predictions, targets)
        return loss + neg


class RhythmNetLoss(nn.Module):
    def __init__(self, weight=100.0):
        super(RhythmNetLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.lambd = weight
        self.gru_outputs_considered = None
        self.custom_loss = RhythmNet_autograd()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, resnet_outputs, gru_outputs, target):
        frame_rate = 25.0
        # resnet_outputs, gru_outputs, _ = outputs
        # target_array = target.repeat(1, resnet_outputs.shape[1])
        l1_loss = self.l1_loss(resnet_outputs, target)
        smooth_loss_component = self.smooth_loss(gru_outputs)

        loss = l1_loss + self.lambd * smooth_loss_component
        return loss

    # Need to write backward pass for this loss function
    def smooth_loss(self, gru_outputs):
        smooth_loss = torch.zeros(1).to(device=self.device)
        self.gru_outputs_considered = gru_outputs.flatten()
        # hr_mean = self.gru_outputs_considered.mean()
        for hr_t in self.gru_outputs_considered:
            # custom_fn = RhythmNet_autograd.apply
            smooth_loss = smooth_loss + self.custom_loss.apply(torch.autograd.Variable(hr_t, requires_grad=True),
                                                               self.gru_outputs_considered,
                                                               self.gru_outputs_considered.shape[0])
        return smooth_loss / self.gru_outputs_considered.shape[0]


class RhythmNet_autograd(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, hr_t, hr_outs, T):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.hr_outs = hr_outs
        ctx.hr_mean = hr_outs.mean()
        ctx.T = T
        ctx.save_for_backward(hr_t)
        # pdb.set_trace()
        # hr_t, hr_mean, T = input

        if hr_t > ctx.hr_mean:
            loss = hr_t - ctx.hr_mean
        else:
            loss = ctx.hr_mean - hr_t

        return loss
        # return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        output = torch.zeros(1).to('cuda')

        hr_t, = ctx.saved_tensors
        hr_outs = ctx.hr_outs

        # create a list of hr_outs without hr_t

        for hr in hr_outs:
            if hr == hr_t:
                pass
            else:
                output = output + (1 / ctx.T) * torch.sign(ctx.hr_mean - hr)

        output = (1 / ctx.T - 1) * torch.sign(ctx.hr_mean - hr_t) + output

        return output, None, None


def Pearson_Loss(predictions, targets):
    rst = 0
    targets = targets[:, :]
    predictions = torch.squeeze(predictions)
    # Pearson correlation can be performed on the premise of normalization of input data
    predictions = (predictions - torch.mean(predictions)) / torch.std(predictions)
    targets = (targets - torch.mean(targets)) / torch.std(targets)

    for i in range(predictions.shape[0]):
        sum_x = torch.sum(predictions[i])  # x
        sum_y = torch.sum(targets[i])  # y
        sum_xy = torch.sum(predictions[i] * targets[i])  # xy
        sum_x2 = torch.sum(torch.pow(predictions[i], 2))  # x^2
        sum_y2 = torch.sum(torch.pow(targets[i], 2))  # y^2
        N = predictions.shape[1] if len(predictions.shape) > 1 else 1
        pearson = (N * sum_xy - sum_x * sum_y) / (
            torch.sqrt((N * sum_x2 - torch.pow(sum_x, 2)) * (N * sum_y2 - torch.pow(sum_y, 2))))

        rst += pearson

    rst = rst / predictions.shape[0]
    return rst


def stft(input_signal):
    stft_sig = torch.stft(input_signal, n_fft=1024, hop_length=512, win_length=1024, window=torch.hamming_window(1024),
                          center=True, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
    return stft_sig

def phase_diff_loss(pred, gt):
    pred_phase = torch.angle(pred)
    gt_phase = torch.angle(gt)
    loss = torch.abs(torch.sum(torch.exp(1j * (pred_phase - gt_phase)))) / pred.size(0)
    return loss



class stftLoss(nn.Module):
    def __init__(self):
        super(stftLoss, self).__init__()

    def forward(self, predictions, targets):
        # targets = targets[:, :]
        # predictions = torch.squeeze(predictions)

        neg = neg_Pearson_Loss(predictions, targets)
        neg_cossim = torch.mean(1.0 - F.cosine_similarity(targets, predictions))
        neg_cossim.requires_grad_(True)
        return neg_cossim + neg


class PearsonLoss(nn.Module):
    def __init__(self):
        super(PearsonLoss, self).__init__()

    def forward(self, predictions, targets):
        return Pearson_Loss(predictions, targets)


def phase_correlation_loss(input, target):
    # Define the forward pass for computing the phase correlation matrix
    def forward(x):
        # Compute the STFTs of the input and target signals
        input_stft = torch.stft(x, n_fft=input.shape[1], window=torch.hann_window(input.shape[1], device=input.device),
                                center=False)
        target_stft = torch.stft(target, n_fft=target.shape[1],
                                 window=torch.hann_window(target.shape[1], device=target.device), center=False)

        # Compute the complex conjugate of the target STFT
        target_conj = torch.conj(target_stft)

        # Compute the phase correlation matrix
        corr_matrix = input_stft * target_conj
        corr_matrix /= torch.abs(corr_matrix)
        corr_matrix = torch.fft.irfft(corr_matrix, dim=1)

        return corr_matrix

    # Compute the phase correlation matrix using memory checkpointing
    corr_matrix = cp.checkpoint(forward, input)

    # Compute the index of the maximum correlation value for each batch element
    max_corr_idx = torch.argmax(corr_matrix, dim=1)

    # Compute the phase correlation coefficient loss for the batch
    loss = 1.0 - torch.mean(torch.cos(torch.tensor(2.0 * np.pi * max_corr_idx / input.shape[1], device=input.device)))

    return loss

def mutual_information_loss(signal1, signal2, num_bins=32):
    # Compute the joint histogram
    hist2d = torch.histc(torch.stack([signal1, signal2], dim=1), bins=num_bins)

    # Compute the marginal histograms
    hist1 = torch.histc(signal1, bins=num_bins)
    hist2 = torch.histc(signal2, bins=num_bins)

    eps = 1e-8
    hist2d = hist2d + eps
    hist1 = hist1 + eps
    hist2 = hist2 + eps

    # Compute the probabilities and entropies
    p12 = hist2d / torch.sum(hist2d)
    p1 = hist1 / torch.sum(hist1)
    p2 = hist2 / torch.sum(hist2)
    H1 = -torch.sum(p1 * torch.log2(p1))
    H2 = -torch.sum(p2 * torch.log2(p2))

    # Compute the mutual information
    MI = torch.sum(p12 * torch.log2(p12 / (torch.outer(p1, p2))))

    # Normalize the mutual information
    NMI = MI / (0.5 * (H1 + H2))

    # Return the negative mutual information as the loss
    return NMI / signal1.shape[0]


def peak_loss(y_true, y_pred, alpha=0.5, beta=1.0):
    def find_peaks_torch(signal, height=None, distance=None):
        signal_np = signal.detach().cpu().numpy()
        peaks, _ = find_peaks(signal_np, height=height, distance=distance)
        return torch.tensor(peaks, dtype=torch.int64)

    def find_peaks_negative(signal, height=None, distance=None):
        signal_np = -1.0 * signal.detach().cpu().numpy()
        peaks, _ = find_peaks(signal_np, height=height, distance=distance)
        return torch.tensor(peaks, dtype=torch.int64)

    def find_peak_freq_torch(signal, fs=30):
        signal_np = signal.detach().cpu().numpy()
        N = _next_power_of_2(signal_np.shape[0])
        f_ppg, pxx_ppg = scipy.signal.periodogram(signal_np, fs=fs, nfft=N, detrend=False)
        fmask_ppg = np.argwhere((f_ppg >= 0.75) & (f_ppg <= 2.5))
        mask_ppg = np.take(f_ppg, fmask_ppg)
        mask_pxx = np.take(pxx_ppg, fmask_ppg)
        peak_freq = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0]
        return peak_freq

    def find_peaks_values(signal, peaks):
        return signal[peaks]

    batch_size = y_true.size(0)
    total_loss = 0

    with torch.no_grad():
        for i in range(batch_size):
            # Find the peaks in the true and predicted signals
            y_true_peaks = find_peaks_torch(y_true[i])
            y_pred_peaks = find_peaks_torch(y_pred[i])

            y_pred_peak_values = find_peaks_values(y_pred[i], y_pred_peaks)

            # Calculate the difference in the number of peaks
            peak_count_difference = np.abs(y_true_peaks.size(0) - y_pred_peaks.size(0))
            peak_value_difference = torch.abs(1 - y_pred_peak_values.mean())

            y_true_peaks = find_peaks_negative(y_true[i])
            y_pred_peaks = find_peaks_negative(y_pred[i])

            y_pred_peak_values = find_peaks_values(y_pred[i], y_pred_peaks)

            # Calculate the difference in the number of peaks
            neg_peak_count_difference = np.abs(y_true_peaks.size(0) - y_pred_peaks.size(0))

            neg_peak_value_difference = torch.abs(1 - y_pred_peak_values.mean())

            y_true_peak_freq = find_peak_freq_torch(y_true[i])
            y_pred_peak_freq = find_peak_freq_torch(y_pred[i])

            # Calculate the difference in peak frequency
            freq_diff = torch.abs(torch.tensor(y_true_peak_freq - y_pred_peak_freq))

            # Combine the losses
            loss = alpha * (peak_count_difference+neg_peak_count_difference + peak_value_difference + neg_peak_value_difference) + freq_diff  # + beta * peak_position_difference
            total_loss += loss

    return total_loss / batch_size

class BVPVelocityLoss(nn.Module):
    def __init__(self):
        super(BVPVelocityLoss, self).__init__()
        self.trip = nn.TripletMarginLoss()
        # a / pos / neg

    def forward(self, predictions, targets):
        # [f,l,r,t]
        # (f >-< t,f <->r) (f >-< t, f<->l)
        # (l >-< t, l <->f) (l >-<r, l <-> f)
        # (r >-< t, r <->f) (r >-<r, r <-> f)

        pearson = neg_Pearson_Loss(predictions, targets)
        # NMI = mutual_information_loss(predictions, targets)
        # phase = phase_correlation_loss(predictions, targets)

        # perd_loss = periodic_signal_loss(targets,predictions)

        loss = pearson + peak_loss(targets,predictions) + derivative_loss(predictions,targets)#+ NMI + phase

        return loss

def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))

# derivative loss for bvp
def derivative_loss(predictions, targets):
    predictions = predictions.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    batch = predictions.shape[0]

    loss = 0

    for i in range(len(predictions)):
        # 1st derivative
        predictions[i] = np.gradient(predictions[i])
        targets[i] = np.gradient(targets[i])
        loss += cos_sim(predictions[i], targets[i])
        # 2nd derivative
        predictions[i] = np.gradient(predictions[i])
        targets[i] = np.gradient(targets[i])
        loss += cos_sim(predictions[i], targets[i])


    return 2 - loss/batch


def periodic_signal_loss(signal, pred_period):

    for s in signal:
        periods = s.unfold(1, pred_period, pred_period).squeeze(0)
        min_values, _ = torch.min(periods, dim=1)
        max_values, _ = torch.max(periods, dim=1)


    # 예측된 주기를 정수로 변환
    pred_period = int(pred_period.item())

    # 주기의 시작 인덱스를 찾기 위해 신호를 여러 개의 주기로 분할
    periods = signal.unfold(1, pred_period, pred_period).squeeze(0)

    # 각 주기의 최소값과 최대값 찾기
    min_values, _ = torch.min(periods, dim=1)
    max_values, _ = torch.max(periods, dim=1)

    # 최소값과 최대값의 차이 계산
    min_max_diff = torch.mean(max_values - min_values)

    return min_max_diff

def autocorrelation(signal, max_lag=None):
    if max_lag is None:
        max_lag = signal.shape[-1] // 2

    acf = torch.zeros(max_lag)
    signal_mean = torch.mean(signal)

    for lag in range(max_lag):
        acf[lag] = torch.mean((signal[:, :-lag - 1] - signal_mean) * (signal[:, lag:] - signal_mean))

    return acf

def bandpass_filter(data, lowcut=0.8, highcut=2.5, fs=30, order=5):
    # Design bandpass filter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    # Apply bandpass filter
    y = sosfiltfilt(sos, data)
    return y