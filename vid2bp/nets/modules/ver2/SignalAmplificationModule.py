import torch
import torch.nn as nn
from vid2bp.nets.modules.sub_modules.Amplitude_module import DBP_module, SBP_module

class SignalAmplifier(nn.Module):
    def __init__(self):
        super(SignalAmplifier, self).__init__()
        self.dbp_module = DBP_module()
        self.sbp_module = SBP_module()

    def forward(self, transformed_signal):
        dbp = torch.abs(self.dbp_module(transformed_signal))
        sbp = torch.abs(self.sbp_module(transformed_signal))
        amplitude = torch.abs(sbp - dbp)

        amplified_signal = torch.mul(transformed_signal, amplitude) + dbp

        return amplified_signal, dbp, sbp
