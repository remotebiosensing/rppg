import torch
import torch.nn as nn
import torch.nn.modules.loss as loss

from log import log_warning


def loss_fn(loss_fn: str = "mse"):
    """
    :param loss_fn: implement loss function for training
    :return: loss function module(class)
    """
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
        return neg_Pearson_Loss(predictions, targets)
