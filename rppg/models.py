import torchinfo
import torchsummary
import time
from log import log_warning, log_info
from nets.models.AxisNet import AxisNet,PhysiologicalGenerator
from nets.models.DeepPhys import DeepPhys
from nets.models.DeepPhys_DA import DeepPhys_DA
from nets.models.PPNet import PPNet
from rppg.nets.PhysNet import PhysNet
from nets.models.PhysNet import PhysNet_2DCNN_LSTM
from nets.models.Seq_GCN import Seq_GCN
from nets.models.TEST import TEST,TEST2,APNET,APNET_Backbone
from nets.models.RhythmNet import RhythmNet
from nets.models.ETArPPGNet import ETArPPGNet
from nets.models.sub_models.VitaMon import Vitamon
from rppg.nets.APNETv2 import APNETv2
from rppg.nets.APNETv3 import APNETv3
from log import log_info_time

import os
NUM_FEATURES = 5
NUM_CLASSES = 10

def get_ver_model(ver:int = 1):
    return TEST2(ver)


def get_model(model_name,time_length):
    """
    :param model_name: model name
    :return: model
    """

    if model_name == "DeepPhys":
        model = DeepPhys()
    elif model_name == "DeepPhys_DA":
        model = DeepPhys_DA()
    elif model_name == "PhysNet":
        model = PhysNet(frames=time_length)
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
    elif model_name == "APNETv2":
        model = APNETv2()
    elif model_name == "APNETv3":
        model = APNETv3()
    else:
        log_warning("pls implemented model")
        raise NotImplementedError("implement a custom model(%s)" % model_name)

    return model.cuda()


def summary(model_name,model):
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
        torchinfo.summary(model, (1, 3, 32, 128, 128))
    elif model_name in "PPNet":
        torchinfo.summary(model, (1, 1, 250))
    elif model_name in "GCN":
        log_warning("some module is not supported in torchinfo")
        torchinfo.summary(model,(32,3,32,32,32))
    else:
        log_warning("use implemented model")
        raise NotImplementedError("implement a custom model(%s) " % model_name)
