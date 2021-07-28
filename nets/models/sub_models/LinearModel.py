import torch

from nets.funcs.complexFunctions import complex_tanh
from nets.layers.complexLayers import ComplexDropout, ComplexLinear


class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.f_drop1 = torch.nn.Dropout(0.25)
        self.f_linear1 = torch.nn.Linear(5184, 256, bias=True)
        self.f_linear2 = torch.nn.Linear(256, 1, bias=True)

    def forward(self, input):
        f1 = torch.flatten(input, start_dim=1)
        f2 = self.f_drop1(f1)
        f3 = torch.tanh(self.f_linear1(f2))
        f4 = self.f_linear2(f3)
        return f4


class ComplexLinearModel(torch.nn.Module):
    def __init__(self):
        super(ComplexLinearModel, self).__init__()
        self.f_drop1 = ComplexDropout(0.25)
        self.f_linear1 = ComplexLinear(64 * 6 * 6, 256)
        self.f_linear2 = ComplexLinear(256, 1)

    def forward(self, input):
        f1 = input.view(-1, 64 * 6 * 6)
        f2 = self.f_drop1(f1)
        f3 = complex_tanh(self.f_linear1(f2))
        f4 = self.f_linear2(f3)
        return f4
