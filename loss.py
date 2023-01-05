import os

# import config
import torch
import torch.nn as nn
import torch.nn.modules.loss as loss
import numpy as np
from log import log_warning
import torch.nn.functional as F
import scipy
import math



from params import params
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
        N = predictions.shape[1]
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
        predictions = torch.fft.fft(predictions, dim=1,norm="forward")
        targets = torch.fft.fft(targets, dim=1,norm="forward")
        loss = loss_func(predictions, targets)
        return loss + neg


class RhythmNetLoss(nn.Module):
    def __init__(self, weight=100.0):
        super(RhythmNetLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.lambd = weight
        self.gru_outputs_considered = None
        self.custom_loss = RhythmNet_autograd()
        self.device = config.DEVICE

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
        output = torch.zeros(1).to(config.DEVICE)

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
        N = predictions.shape[1]
        pearson = (N * sum_xy - sum_x * sum_y) / (
            torch.sqrt((N * sum_x2 - torch.pow(sum_x, 2)) * (N * sum_y2 - torch.pow(sum_y, 2))))

        rst += pearson

    rst = rst / predictions.shape[0]
    return rst

def stft(input_signal):
    stft_sig = torch.stft(input_signal, n_fft=1024, hop_length=512, win_length=1024, window=torch.hamming_window(1024), center=True, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
    return stft_sig



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



def sig_to_BPfiltersig(label,N,fs,hf_low,hf_high):
    '''
    label = gt signal
    N = signal length = len(label)
    fs = fps
    hf_low = low cut off frequency
    hf_high = high cut off frequency
    '''

    # signal->fft signal
    fft_label = scipy.fft.fft(label)

    # fft signal->fft signal shift
    fft_freqs = scipy.fft.fftfreq(N, d=1/fs)

    # bandpass filter
    x_label = len(fft_freqs)
    tmp = fft_freqs[x_label // 2:]
    tmp2 = fft_freqs[:x_label // 2]
    fft_freqs = np.concatenate((tmp, tmp2))

    for i in range(len(fft_freqs)):
        if hf_low <= math.fabs(fft_freqs[i]) <= hf_high:
            continue
        else:
            fft_label[i] = 0

    #bp filter signal -> ifft
    ifft_bp_label = scipy.fft.ifft(fft_label)

    # for i in range(0, len(label), 32):
    #     plt.title('Comapre original signal and Filtered Time signal every 32frames')
    #     plt.plot(label[i:i + 32], label='Ground Truth', color='blue', linewidth=2, alpha=0.5)
    #     plt.plot(ifft_bp_label.real[i:i + 32], label='Filtered', color='red', linewidth=1)
    #     plt.legend()
    #     plt.grid()
    #     plt.show()

    return abs(ifft_bp_label)