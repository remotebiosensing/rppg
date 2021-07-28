from log import log_warning
from nets.models.DeepPhys import DeepPhys
from nets.models.DeepPhys_DA import DeepPhys_DA
from nets.models.PhysNet import PhysNet


def get_model(model_name: str = "DeepPhys"):
    """
    :param model_name: implemented model name
    :return: model
    """
    if model_name == "DeepPhys":
        return DeepPhys()
    elif model_name == "DeepPhys_DA":
        return DeepPhys_DA()
    elif model_name == "PhsNet":
        return PhysNet()
    else:
        log_warning("use implemented model")
        raise NotImplementedError("implement a custom model(%s) in /nets/models/" % model_name)


def is_model_support(model_name, model_list):
    if not (model_name in model_list):
        log_warning("use implemented model")
        raise NotImplementedError("implement a custom model(%s) in /nets/models/" % model_name)
