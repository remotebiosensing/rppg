import os

import torch
import torch.nn as nn
import torch.nn.modules.loss as loss

from log import log_warning

import config


def loss_fn(loss_fn: str = "mse", log_flag: bool = True):
    """
    :param loss_fn: implement loss function for training
    :return: loss function module(class)
    """
    if log_flag:
        print("========= loss_fn() in" + os.path.basename(__file__))

    if loss_fn == "mse":
        return loss.MSELoss()
    elif loss_fn == "L1":
        return loss.L1Loss()
    elif loss_fn == "neg_pearson":
        return NegPearsonLoss()
    elif loss_fn == "multi_margin":
        return loss.MultiMarginLoss()
    elif loss_fn == "bce":
        return loss.BCELoss()
    elif loss_fn == "huber":
        return loss.HuberLoss()
    elif loss_fn == "cosine_embedding":
        return loss.CosineEmbeddingLoss()
    elif loss_fn == "cross_entropy":
        return loss.CrossEntropyLoss()
    elif loss_fn == "ctc":
        return loss.CTCLoss()
    elif loss_fn == "bce_with_logits":
        return loss.BCEWithLogitsLoss()
    elif loss_fn == "gaussian_nll":
        return loss.GaussianNLLLoss()
    elif loss_fn == "hinge_embedding":
        return loss.HingeEmbeddingLoss()
    elif loss_fn == "KLDiv":
        return loss.KLDivLoss()
    elif loss_fn == "margin_ranking":
        return loss.MarginRankingLoss()
    elif loss_fn == "multi_label_margin":
        return loss.MultiLabelMarginLoss()
    elif loss_fn == "multi_label_soft_margin":
        return loss.MultiLabelSoftMarginLoss()
    elif loss_fn == "nll":
        return loss.NLLLoss()
    elif loss_fn == "nll2d":
        return loss.NLLLoss2d()
    elif loss_fn == "pairwise":
        return loss.PairwiseDistance()
    elif loss_fn == "poisson_nll":
        return loss.PoissonNLLLoss()
    elif loss_fn == "smooth_l1":
        return loss.SmoothL1Loss()
    elif loss_fn == "soft_margin":
        return loss.SoftMarginLoss()
    elif loss_fn == "triplet_margin":
        return loss.TripletMarginLoss()
    elif loss_fn == "triplet_margin_distance":
        return loss.TripletMarginWithDistanceLoss()
    elif loss_fn == "RhythmNetLoss":
        return RhythmNetLoss()
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
        return torch.nn.MSELoss(torch.fft.fft(predictions, dim=1), torch.fft.fft(targets, dim=1))

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

        loss = l1_loss + self.lambd*smooth_loss_component
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
                output = output + (1/ctx.T)*torch.sign(ctx.hr_mean - hr)

        output = (1/ctx.T - 1)*torch.sign(ctx.hr_mean - hr_t) + output

        return output, None, None