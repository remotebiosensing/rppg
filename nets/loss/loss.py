import torch
import torch.nn as nn
import numpy as np

'''
np.shape(predictions) = torch.Size([64,7500])
np.shape(targets) = torch.Size([64,7500])
'''


def r(predictions, targets):
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


def rmse_Loss(predictions, targets):
    predictions = torch.squeeze(predictions)
    N = predictions.shape[1]
    global rmse

    for i in range(predictions.shape[0]):
        rmse = torch.sqrt((1 / N) * torch.sum(torch.pow(targets[i] - predictions[i], 2)))
    return rmse


def fft_Loss(predictions, targets):
    predictions = torch.squeeze(predictions)
    # global fft
    rst = 0

    for i in range(predictions.shape[0]):
        rst += torch.nn.MSELoss(torch.fft.fft(predictions[i]), torch.fft.fft(targets[i]))

    rst /= predictions.shape[0]
    return rst


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
