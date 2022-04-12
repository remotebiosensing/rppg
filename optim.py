import torch.optim as opt

from log import log_warning


def optimizer(model_params, learning_rate: float = 1, optim: str = "mse"):
    '''
    call optimizer
    :param model_params: learning target's parameter
    :param learning_rate: learning rate
    :param optim: optimizer
    :return: selected optimizer object
    '''
    if optim == "adam":
        return opt.Adam(model_params, learning_rate)
    elif optim == "sgd":
        return opt.SGD(model_params, learning_rate)
    elif optim == "rms_prop":
        return opt.RMSprop(model_params, learning_rate)
    elif optim == "ada_delta":
        return opt.Adadelta(model_params, learning_rate)
    elif optim == "ada_grad":
        return opt.Adagrad(model_params, learning_rate)
    elif optim == "ada_max":
        return opt.Adamax(model_params, learning_rate)
    elif optim == "ada_mw":
        return opt.AdamW(model_params, learning_rate,weight_decay=0.9)
    elif optim == "a_sgd":
        return opt.ASGD(model_params, learning_rate)
    elif optim == "lbfgs":
        return opt.LBFGS(model_params, learning_rate)
    elif optim == "n_adam":
        return opt.NAdam(model_params, learning_rate)
    elif optim == "r_adam":
        return opt.RAdam(model_params, learning_rate,betas=(0.9,0.999),eps=1e-08,weight_decay=1e-2)
    elif optim == "rprop":
        return opt.Rprop(model_params, learning_rate)
    elif optim == "sparse_adam":
        return opt.SparseAdam(model_params, learning_rate)
    else:
        log_warning("use implemented optimizer")
        raise NotImplementedError("implement a custom optimizer(%s) in optimizer.py" % optim)
