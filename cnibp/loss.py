import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cnibp.preprocessing.utils.signal_utils as su
from scipy import signal
from torchmetrics.functional import mean_absolute_percentage_error as mape
from torchmetrics.functional import symmetric_mean_absolute_percentage_error as smape


# TODO : '''jaccard similarity'''

# import torch
# import torch.nn.functional as F

def angle_loss(predictions, targets):
    rst = 0
    # mseLoss = nn.MSELoss()
    predictions_angle = torch.angle(predictions) * 180 / torch.pi
    # predictions_angle = predictions_angle*180/torch.pi
    targets_angle = torch.angle(targets) * 180 / torch.pi

    for p, t in zip(predictions_angle, targets_angle):
        rst += smape(p, t) / 2
    rst /= predictions.shape[0]

    return rst


def mutual_information_loss(signal1, signal2, num_bins=256):
    # Compute the joint histogram
    hist2d, _, _ = torch.histc(torch.stack([signal1, signal2], dim=1), bins=num_bins)

    # Compute the marginal histograms
    hist1, _ = torch.histc(signal1, bins=num_bins)
    hist2, _ = torch.histc(signal2, bins=num_bins)

    # Compute the probabilities and entropies
    p12 = hist2d / torch.sum(hist2d)
    p1 = hist1 / torch.sum(hist1)
    p2 = hist2 / torch.sum(hist2)
    H1 = -torch.sum(p1 * torch.log2(p1 + 1e-8))
    H2 = -torch.sum(p2 * torch.log2(p2 + 1e-8))

    # Compute the mutual information
    MI = torch.sum(p12 * torch.log2(p12 / (torch.ger(p1, p2) + 1e-8)))

    # Normalize the mutual information
    NMI = MI / (0.5 * (H1 + H2))

    # Return the negative mutual information as the loss
    return -NMI


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


def stft_similarity_loss(predictions, targets):
    # predictions = torch.squeeze(predictions)
    # cos_sim0 = nn.CosineSimilarity(dim=0, eps=1e-6)
    # cos_sim2 = nn.CosineSimilarity(dim=2, eps=1e-6)
    # cos_sim3 = nn.CosineSimilarity(dim=3, eps=1e-6)
    cos_sim1 = nn.CosineSimilarity(dim=1, eps=1e-6)
    win = torch.hann_window(60).to('cuda:0')
    pred = torch.stft(predictions, n_fft=60, hop_length=60 // 4, window=win, return_complex=True)
    tar = torch.stft(targets, n_fft=60, hop_length=60 // 4, window=win, return_complex=True)
    # for i in range(predictions.shape[0]):
    #     Zxx_pred = torch.stft(predictions[i], n_fft=60, hop_length=None)
    #     Zxx_tar = torch.stft(targets[i], n_fft=60, hop_length=None)
    # f_pred, t_pred, Zxx_pred = signal.stft(predictions[i], fs=60, nperseg=100)
    # f_tar, t_tar, Zxx_tar = signal.stft(targets[i], fs=60, nperseg=100)
    # test1 = cos_sim(Zxx_pred, Zxx_tar)
    # test2 = cos_sim2(Zxx_pred, Zxx_tar)
    # rst += 1 - cos_sim(Zxx_pred, Zxx_tar)

    # rst = rst / predictions.shape[0]
    return 1 - torch.mean(cos_sim1(abs(pred), abs(tar)))


def scale_loss(dbp, sbp):
    return 1. - (torch.sum(dbp < sbp) / dbp.shape[0])


# def rmsse_loss(targets)

def amplitude_loss(dbp_predictions, sbp_predictions, amplitude_predictions, dbp_targets, sbp_targets, mbp_targets):
    scale_cost = 1. - torch.sum(dbp_predictions < sbp_predictions) / dbp_predictions.shape[0]
    dbp_cov_mat = torch.cov(torch.stack((dbp_targets.view(-1, 1), dbp_predictions), dim=1).squeeze().T)
    sbp_cov_mat = torch.cov(torch.stack((sbp_targets.view(-1, 1), sbp_predictions), dim=1).squeeze().T)
    dbp_eig_val = torch.linalg.eigvals(dbp_cov_mat)
    sbp_eig_val = torch.linalg.eigvals(sbp_cov_mat)
    dbp_eig1, dbp_eig2 = dbp_eig_val[0], dbp_eig_val[-1]
    sbp_eig1, sbp_eig2 = sbp_eig_val[0], sbp_eig_val[-1]
    # amp_target = sbp_targets - dbp_targets
    batchN = sbp_targets.shape[0]

    dbp_predictions = torch.squeeze(dbp_predictions)
    sbp_predictions = torch.squeeze(sbp_predictions)
    rst_d = 0
    rst_s = 0
    normal_mask = sbp_targets.less(120) & dbp_targets.less(80)
    elevated_mask = ((sbp_targets.ge(120) & sbp_targets.less(130)) & dbp_targets.less(80)) \
        # & ~normal_mask
    hyper1_mask = ((sbp_targets.ge(130) & sbp_targets.less(140)) | (dbp_targets.ge(80) & dbp_targets.less(90))) \
        # & ~(normal_mask | elevated_mask)
    hyper2_mask = (sbp_targets.ge(140) | dbp_targets.ge(90)) \
        # & ~(normal_mask | elevated_mask | hyper1_mask)
    hyper_crisis_mask = (sbp_targets.greater(180) | dbp_targets.greater(120))  #
    # & ~(normal_mask|elevated_mask|hyper1_mask|hyper2_mask)

    mask_list = [normal_mask, elevated_mask, hyper1_mask, hyper2_mask, hyper_crisis_mask]
    # mask_size = [torch.sum(m) for m in mask_list]
    mask_weight_dbp = [torch.sqrt(torch.log(batchN / (torch.sum(m)))) for m in mask_list]
    mask_weight_sbp = [torch.sqrt(torch.log(batchN / (torch.sum(m)))) for m in mask_list]
    # mask_x_expnegx = [torch.sum(m)*torch.exp(-torch.sum(m)) for m in mask_list]
    # mask_lnxoverx = [torch.log(torch.sum(m))/torch.sum(m) for m in mask_list]
    # mask_lnxoverxplusx = [torch.log(torch.sum(m))/torch.sum(m)+torch.sum(m) for m in mask_list]
    # mask_weight = [torch.sqrt(torch.log(batchN/torch.sum(hypo_mask))), torch.sqrt(torch.log(batchN/torch.sum(normal_mask))),
    #                torch.sqrt(torch.log(batchN/torch.sum(hyper1_mask))), torch.sqrt(torch.log(batchN/torch.sum(hyper2_mask))),
    #                torch.sqrt(torch.log(batchN/torch.sum(hyper_crisis_mask)))]
    # m_rst = 0
    mask_cnt = 0
    s_rst = 0
    d_rst = 0
    for m, w_s, w_d in zip(mask_list, mask_weight_sbp, mask_weight_dbp):
        masked_dbp_pred, masked_dbp_tar = dbp_predictions.masked_select(m), dbp_targets.masked_select(m)
        masked_sbp_pred, masked_sbp_tar = sbp_predictions.masked_select(m), sbp_targets.masked_select(m)
        # masked_mbp_pred, masked_mbp_tar = amplitude_predictions.masked_select(m), amp_target.masked_select(m)
        if int(torch.sum(m)) != 0:
            mask_cnt += 1
            for i in range(int(torch.sum(m))):
                d_rst += smape(masked_dbp_pred[i], masked_dbp_tar[i]) * w_d
                s_rst += smape(masked_sbp_pred[i], masked_sbp_tar[i]) * w_s
                # m_rst += smape(masked_mbp_pred[i], masked_mbp_tar[i]) * w
            d_rst /= int(torch.sum(m))
            s_rst /= int(torch.sum(m))
            # m_rst /= 2
        else:
            continue
        rst_d += d_rst
        rst_s += s_rst
    if mask_cnt == 0:
        rst_d /= 5
        rst_s /= 5
    else:
        rst_d /= mask_cnt
        rst_s /= mask_cnt
    # dbp_pca = torch.abs(dbp_eig1.real - dbp_eig2.real) / dbp_eig1.real
    # sbp_pca = torch.abs(sbp_eig1.real - sbp_eig2.real) / sbp_eig1.real
    dbp_mean_cost = torch.abs(torch.mean(dbp_targets) - torch.mean(dbp_predictions)) / torch.mean(dbp_targets)
    sbp_mean_cost = torch.abs(torch.mean(sbp_targets) - torch.mean(sbp_predictions)) / torch.mean(sbp_targets)
    dbp_var_cost = torch.abs(torch.var(dbp_targets) - torch.var(dbp_predictions)) / torch.var(dbp_targets)
    sbp_var_cost = torch.abs(torch.var(sbp_targets) - torch.var(sbp_predictions)) / torch.var(sbp_targets)
    return rst_d + dbp_mean_cost + dbp_var_cost, rst_s + sbp_mean_cost + sbp_var_cost, scale_cost  # + (1 + eig2.imag / eig2.real)


def Systolic_Loss(predictions, targets):
    predictions = torch.squeeze(predictions)
    targets = torch.squeeze(targets)
    rst = 0
    for i in range(predictions.shape[0]):
        rst += torch.abs((targets[i] - predictions[i]) / targets[i])
    rst /= predictions.shape[0]
    return rst


def Diastolic_Loss(predictions, targets):
    predictions = torch.squeeze(predictions)
    targets = torch.squeeze(targets)
    rst = 0
    for i in range(predictions.shape[0]):
        rst += torch.abs((targets[i] - predictions[i]) / targets[i])
    rst /= predictions.shape[0]
    return rst


def amp_loss(amp_predictions, dbp_targets, sbp_targets):
    amp_targets = sbp_targets - dbp_targets
    rst = 0.
    for i in range(amp_targets.shape[0]):
        rst += smape(amp_predictions[i], amp_targets[i]) / 2
    rst /= amp_targets.shape[0]

    return rst


def mean_average_percentage_error_loss(predictions, targets):
    rst = 0
    predictions = torch.squeeze(predictions)
    for i in range(predictions.shape[0]):
        rst += mape(predictions[i], targets[i])

    rst /= predictions.shape[0]

    return rst


def rmse_Loss(predictions, targets):
    # predictions = torch.squeeze(predictions)
    N = predictions.shape[-1]
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

def Self_Scale_Loss(scaled_ple, _):
    scaled_ple = scaled_ple.view(scaled_ple.shape[0], -1)
    # sbp_var = torch.var(scaled_ple[:, 0])

    sbp_var = torch.std(torch.max(scaled_ple, dim=1).values)
    dbp_var = torch.std(torch.min(scaled_ple, dim=1).values)

    rst = torch.abs(sbp_var + dbp_var)

    return rst


def Scaled_Vector_Cos_Sim_Loss(predictions, targets):
    pred_max = torch.max(predictions, dim=-1, keepdim=True).values
    pred_min = torch.min(predictions, dim=-1, keepdim=True).values
    pred_min_max_vector = torch.cat((pred_min, pred_max), dim=-1).view(-1, 1, 2)
    pred_mean_std_vector = torch.cat((torch.mean(predictions, dim=-1, keepdim=True),
                                      torch.std(predictions, dim=-1, keepdim=True)), dim=-1).view(-1, 1, 2)
    # pred_vec = (predictions - pred_min[0]) / (pred_max[0] - pred_min[0])
    # pred_start_point = torch.cat((pred_min, pred_min), dim=-1).view(-1, 1, 2)
    # pred_end_point = torch.cat((pred_max, pred_max), dim=-1).view(-1, 1, 2)
    pred_vector = torch.cat((pred_min_max_vector, pred_mean_std_vector), dim=1)
    tar_max = torch.max(targets, dim=-1, keepdim=True).values
    tar_min = torch.min(targets, dim=-1, keepdim=True).values
    tar_min_max_vector = torch.cat((tar_min, tar_max), dim=-1).view(-1, 1, 2)
    tar_mean_std_vector = torch.cat((torch.mean(targets, dim=-1, keepdim=True),
                                     torch.std(targets, dim=-1, keepdim=True)),
                                    dim=-1).view(-1, 1, 2)
    # tar_vec = (targets - tar_min[0]) / (tar_max[0] - tar_min[0])
    # tar_start_point = torch.cat((tar_min, tar_min), dim=-1).view(-1, 1, 2)
    # tar_end_point = torch.cat((tar_max, tar_max), dim=-1).view(-1, 1, 2)
    tar_vector = torch.cat((tar_min_max_vector, tar_mean_std_vector), dim=1)
    cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-6)

    # return 1 - torch.mean(cos_sim(pred_vector, tar_vector))
    return 1 - torch.mean(cos_sim(predictions, targets))
    # for i in range(predictions.shape[0]):
    #     cos_s1 = cos_sim(pred_vector[i], tar_vector[i])
    #     cos_s2 = cos_sim(pred_test[i], tar_test[i])
    # rst += 2 - (torch.mean(cos_s1) + torch.mean(cos_s2))
    # rst /= predictions.shape[0]
    # return rst


def Neg_Pearson_Loss(predictions, targets):
    # print('Neg*** prediction.shape :', np.shape(predictions), 'targets.shape :', np.shape(targets))
    '''
    :param predictions: inference value of trained model
    :param targets: target label of input data
    :return: negative pearson loss
    '''
    rst = 0

    '''
    Pearson correlation can be performed on the premise of normalization of input data
    '''
    # predictions = (predictions - torch.mean(predictions)) / torch.std(predictions)
    # targets = (targets - torch.mean(targets)) / torch.std(targets)
    predictions = (predictions - torch.mean(predictions, dim=-1, keepdim=True)) / torch.std(predictions, dim=-1,
                                                                                            keepdim=True)
    targets = (targets - torch.mean(targets, dim=-1, keepdim=True)) / torch.std(targets, dim=-1, keepdim=True)

    # masked_predictions = (targets - predictions).masked_select(targets == 0)
    # masked_predictions_test = (test_targets - test_predictions).masked_select(test_targets == 0.)
    # false_mask = (targets - predictions).greater(0.5)
    # true_mask = (targets - predictions).le(0.)
    # false_masked_prediction_test1 = test_predictions.masked_select(false_mask)
    # true_masked_prediction_test1 = test_predictions.masked_select(true_mask)
    # for i in range(predictions.shape[0]):
    #     # predictions[i] = predictions[i] - torch.mean(predictions[i])
    #     # targets[i] = targets[i] - torch.mean(targets[i])
    #     sum_x = torch.sum(predictions[i])  # x
    #     sum_y = torch.sum(targets[i])  # y
    #     sum_xy = torch.sum(predictions[i] * targets[i])  # xy
    #     sum_x2 = torch.sum(torch.pow(predictions[i], 2))  # x^2
    #     sum_y2 = torch.sum(torch.pow(targets[i], 2))  # y^2
    #     N = predictions.shape[1]
    #     pearson = (N * sum_xy - sum_x * sum_y) / (
    #         torch.sqrt((N * sum_x2 - torch.pow(sum_x, 2)) * (N * sum_y2 - torch.pow(sum_y, 2)))) + eps
    #
    #     rst += 1 - pearson

    # targets = targets[:,:]
    # print('before squeeze', predictions.shape)
    # predictions = predictions.view(predictions.shape[0], -1)
    # predictions = torch.squeeze(predictions, 1)
    # print('after squeeze', predictions.shape)
    # '''
    # Pearson correlation can be performed on the premise of normalization of input data
    # '''
    # predictions = (predictions - torch.mean(predictions)) / torch.std(predictions)
    # targets = (targets - torch.mean(targets)) / torch.std(targets)

    # ''' by definition of Pearson correlation '''
    # torch.mean(torch.cov(torch.stack((predictions[0], targets[0]))) / torch.std(targets[0]) * torch.std(predictions[0]))
    # for i in range(predictions.shape[0]):
    #     cov = torch.mean(torch.cov(torch.stack((predictions[i], targets[i]))))
    #     std_pred = torch.std(predictions[i])
    #     std_tar = torch.std(targets[i])
    #     rst += 1 - cov / (std_pred * std_tar + eps)

    # for i in range(predictions.shape[0]):
    #     pearson = torch.corrcoef(torch.stack((predictions[i], targets[i])))[0][1]
    #     rst += 1 - pearson
    # for i in range(predictions.shape[0]):
    #     # sum_x = torch.sum(predictions[i]-torch.mean(predictions[i]))
    #     # sum_y = torch.sum(targets[i]-torch.mean(targets[i]))
    #     sum_xy = torch.sum((predictions[i]-torch.mean(predictions[i])) * (targets[i]-torch.mean(targets[i])))
    #     pow_x = torch.sum(torch.pow(predictions[i]-torch.mean(predictions[i]), 2))
    #     pow_y = torch.sum(torch.pow(targets[i]-torch.mean(targets[i]), 2))
    #     pearson = (sum_xy / torch.sqrt(pow_x * pow_y))
    #
    #     rst += 1-pearson
    #

    for i in range(predictions.shape[0]):
        # predictions[i] = (predictions[i] - torch.mean(predictions[i])) / torch.std(predictions[i])
        # targets[i] = (targets[i] - torch.mean(targets[i])) / torch.std(targets[i])
        sum_x = torch.sum(predictions[i])  # x
        sum_y = torch.sum(targets[i])  # y
        sum_xy = torch.sum(torch.mul(predictions[i], targets[i]))  # xy
        sum_x2 = torch.sum(torch.pow(predictions[i], 2))  # x^2
        sum_y2 = torch.sum(torch.pow(targets[i], 2))  # y^2
        N = predictions.shape[1]
        pearson = (N * sum_xy - sum_x * sum_y) / (
            torch.sqrt((N * sum_x2 - torch.pow(sum_x, 2)) * (N * sum_y2 - torch.pow(sum_y, 2))))
        # if torch.isnan(pearson):
        # print('pearson is nan')
        # print('N :', N, 'sum_xy :', sum_xy, 'sum_x :', sum_x, 'sum_y :', sum_y, 'sum_x2 :', sum_x2, 'sum_y2 :',
        #       sum_y2)
        # pearson = 0
        rst += 1 - pearson
    # n = predictions.shape[0]
    # sum_x = torch.sum(predictions, dim=1)
    # sum_y = torch.sum(targets, dim=1)
    # sum_xy = torch.sum(torch.mul(predictions-torch.mean(predictions), targets-torch.mean(targets)), dim=1)
    # sum_x_square = torch.sum(torch.pow(predictions-torch.mean(predictions), 2), dim=1)
    # sum_y_square = torch.sum(torch.pow(targets-torch.mean(targets), 2), dim=1)
    # pearson = (sum_xy / (torch.sqrt(sum_x_square * sum_y_square))).mean()

    rst = rst / predictions.shape[0]
    return rst


class NegPearsonLoss(nn.Module):
    def __init__(self):
        super(NegPearsonLoss, self).__init__()

    def forward(self, predictions, targets):
        return Neg_Pearson_Loss(predictions, targets)


class ScaledVectorCosineSimilarity(nn.Module):
    def __init__(self):
        super(ScaledVectorCosineSimilarity, self).__init__()

    def forward(self, predictions, targets):
        return Scaled_Vector_Cos_Sim_Loss(predictions, targets)


class FFTLoss(nn.Module):
    def __init__(self):
        super(FFTLoss, self).__init__()

    def forward(self, predictions, targets):
        return fft_Loss(predictions=predictions, targets=targets)


class STFTLoss(nn.Module):
    def __init__(self):
        super(STFTLoss, self).__init__()

    def forward(self, predictions, targets):
        return stft_similarity_loss(predictions=predictions, targets=targets)


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, predictions, targets):
        return rmse_Loss(predictions=predictions, targets=targets)


class SBPLoss(nn.Module):
    def __init__(self):
        super(SBPLoss, self).__init__()

    def forward(self, predictions, targets):
        return Systolic_Loss(predictions, targets)


class DBPLoss(nn.Module):
    def __init__(self):
        super(DBPLoss, self).__init__()

    def forward(self, predictions, targets):
        return Diastolic_Loss(predictions, targets)


class SelfScaler(nn.Module):
    def __init__(self):
        super(SelfScaler, self).__init__()
        # self.sbp_std = nn.Parameter(torch.tensor(1.0))
        # self.dbp_std = nn.Parameter(torch.tensor(1.0))

    def forward(self, predictions, _):
        return Self_Scale_Loss(predictions, _)


class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, predictions, targets):
        return mean_average_percentage_error_loss(predictions, targets)


class ScaleLoss(nn.Module):
    def __init__(self):
        super(ScaleLoss, self).__init__()

    def forward(self, dbp, sbp):
        return scale_loss(dbp, sbp)


class AmpLoss(nn.Module):
    def __init__(self):
        super(AmpLoss, self).__init__()

    def forward(self, dbp_pred, sbp_pred, mbp_pred, d, s, m):
        return amplitude_loss(dbp_pred, sbp_pred, mbp_pred, d, s, m)


class ALoss(nn.Module):
    def __init__(self):
        super(ALoss, self).__init__()

    def forward(self, amp_pred, d_tar, s_tar):
        return amp_loss(amp_pred, d_tar, s_tar)


class AngleLoss(nn.Module):

    def __init__(self):
        super(AngleLoss, self).__init__()

    def forward(self, pred, tar):
        return angle_loss(pred, tar)
