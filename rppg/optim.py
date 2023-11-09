import torch.optim as opt

from log import log_warning


def optimizer(model_params, learning_rate: float = 1, optim: str = "mse", log_flag: bool = True):
    '''
    call optimizer
    :param model_params: learning target's parameter
    :param learning_rate: learning rate
    :param optim: optimizer
    :return: selected optimizer object
    '''

    if optim == "Adam":
        return opt.Adam(model_params, learning_rate, weight_decay=5e-5)
    elif optim == "SGD":
        return opt.SGD(model_params, learning_rate)
    elif optim == "RMSprop":
        return opt.RMSprop(model_params, learning_rate)
    elif optim == "Adadelta":
        return opt.Adadelta(model_params, learning_rate)
    elif optim == "Adagrad":
        return opt.Adagrad(model_params, learning_rate)
    elif optim == "Adamax":
        return opt.Adamax(model_params, learning_rate)
    elif optim == "AdamW":
        return opt.AdamW(model_params, learning_rate)
    elif optim == "ASGD":
        return opt.ASGD(model_params, learning_rate)
    elif optim == "LBFGS":
        return opt.LBFGS(model_params, learning_rate)
    elif optim == "NAdam":
        return opt.NAdam(model_params, learning_rate)
    elif optim == "RAdam":
        return opt.RAdam(model_params, learning_rate)
    elif optim == "Rprop":
        return opt.Rprop(model_params, learning_rate)
    elif optim == "SparseAdam":
        return opt.SparseAdam(model_params, learning_rate)
    else:
        log_warning("use implemented optimizer")
        raise NotImplementedError("implement a custom optimizer(%s) in optimizer.py" % optim)
