import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import vid2bp.preprocessing.utils.signal_utils as su

# TODO : '''jaccard similarity'''



def fft_Loss(predictions, targets):
    predictions = torch.squeeze(predictions)
    mseLoss = nn.MSELoss()
    rst = 0
    for p, t in zip(predictions, targets):

        # p_fft = torch.fft.fft(p, norm='forward')
        # t_fft = torch.fft.fft(t, norm='forward')
        # rst += torch.sqrt(mseLoss(su.get_systolic(p.detach().cpu()), su.get_systolic(t.detach().cpu())))
        # rst += torch.sqrt(mseLoss(su.get_diastolic(p.detach().cpu()), su.get_diastolic(t.detach().cpu())))
        rst += torch.sqrt(mseLoss(p, t))
        # rst += mseLoss(p, t)
        # rst += torch.sqrt(mseLoss((abs(p_fft) * (2 / len(p_fft)))[:180], (abs(t_fft) * (2 / len(t_fft)))[:180]))
    rst /= predictions.shape[0]
    # rst += Neg_Pearson_Loss(predictions, targets)
    return rst

    # for i in range(predictions.shape[0]):
    #     dc_removed_predictions = predictions[i] - torch.mean(predictions[i])
    #     de_removed_targets = targets[i] - torch.mean(targets[i])
    #     # rst += torch.nn.MSELoss(torch.fft.fft(predictions[i]), torch.fft.fft(targets[i]))
    #     p_fft = torch.fft.fft(dc_removed_predictions)
    #     t_fft = torch.fft.fft(de_removed_targets)
    #     # rst += torch.nn.MSELoss(abs(p_fft) * (2 / len(p_fft)), abs(t_fft) * (2 / len(t_fft)))
    #     # rst += torch.pow(torch.sum(abs(p_fft) * (2 / len(p_fft)) - torch.abs(t_fft) * (2 / len(t_fft))), 2)
    #     mseLoss = nn.MSELoss()
    #     rst += torch.sqrt(mseLoss(abs(p_fft) * (2 / len(p_fft)), abs(t_fft) * (2 / len(t_fft))))
    # rst /= predictions.shape[0]
    # return rst


def Systolic_Loss(predictions, targets):
    predictions = torch.squeeze(predictions)
    rst = 0
    for i in range(predictions.shape[0]):
        rst += torch.abs((targets[i] - predictions[i]) / targets[i])
    rst /= predictions.shape[0]
    return rst


def Diastolic_Loss(predictions, targets):
    predictions = torch.squeeze(predictions)
    rst = 0
    for i in range(predictions.shape[0]):
        rst += torch.abs((targets[i] - predictions[i]) / targets[i])
    rst /= predictions.shape[0]
    return rst


def rmse_Loss(predictions, targets):
    predictions = torch.squeeze(predictions)
    N = predictions.shape[1]
    global rmse

    for i in range(predictions.shape[0]):
        rmse = torch.sqrt((1 / N) * torch.sum(torch.pow(targets[i] - predictions[i], 2)))
    return rmse


# def fft_Loss(predictions, targets):
#     predictions = torch.squeeze(predictions)
#     # global fft
#     rst = 0
#
#     for i in range(predictions.shape[0]):
#         rst += torch.nn.MSELoss(torch.fft.fft(predictions[i]), torch.fft.fft(targets[i]))
#
#     rst /= predictions.shape[0]
#     return rst


def Neg_Pearson_Loss(predictions, targets):
    # print('Neg*** prediction.shape :', np.shape(predictions), 'targets.shape :', np.shape(targets))
    '''
    :param predictions: inference value of trained model
    :param targets: target label of input data
    :return: negative pearson loss
    '''
    rst = 0
    # targets = targets[:,:]
    # print('before squeeze', predictions.shape)
    predictions = torch.squeeze(predictions, 1)
    # print('after squeeze', predictions.shape)
    '''
    Pearson correlation can be performed on the premise of normalization of input data
    '''
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


class NegPearsonLoss(nn.Module):
    def __init__(self):
        super(NegPearsonLoss, self).__init__()

    def forward(self, predictions, targets):
        return Neg_Pearson_Loss(predictions, targets)


class fftLoss(nn.Module):
    def __init__(self):
        super(fftLoss, self).__init__()

    def forward(self, predictions, targets):
        return fft_Loss(predictions=predictions, targets=targets)


class rmseLoss(nn.Module):
    def __init__(self):
        super(rmseLoss, self).__init__()

    def forward(self, predictions, targets):
        return rmse_Loss(predictions=predictions, targets=targets)


class sbpLoss(nn.Module):
    def __init__(self):
        super(sbpLoss, self).__init__()

    def forward(self, predictions, targets):
        return Systolic_Loss(predictions, targets)


class dbpLoss(nn.Module):
    def __init__(self):
        super(dbpLoss, self).__init__()

    def forward(self, predictions, targets):
        return Diastolic_Loss(predictions, targets)
