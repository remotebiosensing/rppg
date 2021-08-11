import torchinfo
import torchsummary

from log import log_warning, log_info
from nets.models.DeepPhys import DeepPhys
from nets.models.DeepPhys_DA import DeepPhys_DA
from nets.models.PhysNet import PhysNet
from nets.models.PhysNet import PhysNet_2DCNN_LSTM


def get_model(model_name: str = "DeepPhys"):
    """
    :param model_name: model name
    :return: model
    """
    if model_name == "DeepPhys":
        return DeepPhys()
    elif model_name == "DeepPhys_DA":
        return DeepPhys_DA()
    elif model_name == "PhysNet":
        return PhysNet()
    elif model_name == "PhysNet_LSTM":
        return PhysNet_2DCNN_LSTM()
    else:
        log_warning("use implemented model")
        raise NotImplementedError("implement a custom model(%s) in /nets/models/" % model_name)


def is_model_support(model_name, model_list):
    """
    :param model_name: model name
    :param model_list: implemented model list
    :return: model
    """
    if not (model_name in model_list):
        log_warning("use implemented model")
        raise NotImplementedError("implement a custom model(%s) in /nets/models/" % model_name)


def summary(model, model_name):
    """
    :param model: torch.nn.module class
    :param model_name: implemented model name
    :return: model
    """
    log_info("=========================================")
    log_info(model_name)
    log_info("=========================================")
    if model_name == "DeepPhys" or model_name == DeepPhys_DA:
        torchsummary.summary(model, (2, 3, 36, 36))
    elif model_name == "PhysNet" or model_name == "PhysNet_LSTM":
        #Use torchinfo support recursive layers(RNN,LSTM)
        #torchinfo: (model_name, input_size=(batch_size, input_size))
        torchinfo.summary(model,(1, 3, 32, 128, 128))
    else:
        log_warning("use implemented model")
        raise NotImplementedError("implement a custom model(%s) in /nets/models/" % model_name)
