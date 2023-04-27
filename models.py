import torchinfo
import torchsummary
import time
from log import log_warning, log_info
from nets.models.AxisNet import AxisNet,PhysiologicalGenerator
from nets.models.DeepPhys import DeepPhys
from nets.models.DeepPhys_DA import DeepPhys_DA
from nets.models.PPNet import PPNet
from nets.models.PhysNet import PhysNet
from nets.models.PhysNet import PhysNet_2DCNN_LSTM
from nets.models.Seq_GCN import Seq_GCN
from nets.models.TEST import TEST,TEST2,APNET,APNET_Backbone
from nets.models.RhythmNet import RhythmNet
from nets.models.ETArPPGNet import ETArPPGNet
from nets.models.sub_models.VitaMon import Vitamon
from log import log_info_time
from params import params

import os
NUM_FEATURES = 5
NUM_CLASSES = 10

def get_ver_model(ver:int = 1):
    return TEST2(ver)


def get_model(model_name):
    """
    :param model_name: model name
    :return: model
    """

    if model_name == "DeepPhys":
        model = DeepPhys()
    elif model_name == "DeepPhys_DA":
        model = DeepPhys_DA()
    elif model_name == "PhysNet":
        model = PhysNet()
    elif model_name == "PhysNet_LSTM":
        model = PhysNet_2DCNN_LSTM()
    elif model_name == "PPNet":
        model = PPNet()
    elif model_name == "GCN":
        model = TEST()#Seq_GCN()#TEST()#
    elif model_name == "AxisNet":
        model = AxisNet(),PhysiologicalGenerator()
    elif model_name == "RhythmNet":
        model = RhythmNet()
    elif model_name == "ETArPPGNet":
        model = ETArPPGNet()
    elif model_name == "Vitamon":
        model =  Vitamon()
    elif model_name =="TEST":
        model = APNET()
    else:
        log_warning("pls implemented model")
        raise NotImplementedError("implement a custom model(%s) in /nets/models/" % params.model)

    # if params.__MODEL_SUMMARY__:
    #     summary(model)

    return model.cuda()


def is_model_support():
    """
    :param model_name: model name
    :param model_list: implemented model list
    :return: model
    """
    if params.log_flag:
        print("========= model support check is model support() in " + os.path.basename(__file__))
    if not (params.model in params.model_list):
        log_warning("use implemented model")
        raise NotImplementedError("implement a custom model(%s) in /nets/models/" % model_name)


def summary(model):
    """
    :param model: torch.nn.module class
    :param model_name: implemented model name
    :return: model
    """
    log_info("=========================================")
    log_info(params.model)
    log_info("=========================================")
    if params.model == "DeepPhys" or params.model == DeepPhys_DA:
        torchsummary.summary(model, (2, 3, 36, 36))
    elif params.model == "PhysNet" or params.model == "PhysNet_LSTM":
        torchinfo.summary(model, (1, 3, 32, 128, 128))
    elif params.model in "PPNet":
        torchinfo.summary(model, (1, 1, 250))
    elif params.model in "GCN":
        log_warning("some module is not supported in torchinfo")
        torchinfo.summary(model,(32,3,32,32,32))
    else:
        log_warning("use implemented model")
        raise NotImplementedError("implement a custom model(%s) in /nets/models/" % model_name)
