import torch
from torch import nn

import torch.nn.functional as F
class TSM(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self,input, n_frame=4, fold_div=3):
        n_frame=4
        B, C, H, W = input.shape
        input = input.view(-1,n_frame,H,W,C)
        fold = C // fold_div
        last_fold = C - (fold_div - 1) * fold
        out1, out2, out3 = torch.split(input,[fold,fold,last_fold],-1)

        padding1 = torch.zeros_like(out1)
        padding1 = padding1[:,-1,:,:,:]
        padding1 = torch.unsqueeze(padding1,1)
        _, out1 = torch.split(out1,[1,n_frame -1], 1)
        out1 = torch.cat((out1,padding1),1)

        padding2 = torch.zeros_like(out2)
        padding2 = padding2[:,0,:,:,:]
        padding2 = torch.unsqueeze(padding2,1)
        out2, _ = torch.split(out2,[n_frame -1,1], 1)
        out2 = torch.cat((padding2,out2),1)

        out = torch.cat((out1,out2,out3), -1)
        out = out.view([-1,C,H,W])

        return out

class TSM_block(torch.nn.Module):
    def __init__(self,in_channels,out_channels, kernel_size):
        super().__init__()
        self.tsm1 = TSM()
        self.t_conv1 = torch.nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size=kernel_size,padding=1)

    def forward(self,input,n_frame=2, fold_div=3):
        t = self.tsm1(input,n_frame,fold_div)
        t = self.t_conv1(t)
        return t

class appearnce_block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        # layer 1
        self.a_conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       stride=1, padding=1)
        self.a_batch1 = torch.nn.BatchNorm2d(self.a_conv1.out_channels)
        # layer 2
        self.a_conv2 = torch.nn.Conv2d(in_channels=self.a_conv1.out_channels, out_channels=out_channels,
                                       kernel_size=kernel_size, stride=1, padding=1)
        self.a_batch2 = torch.nn.BatchNorm2d(self.a_conv2.out_channels)

    def forward(self, inputs):
        # layer 1
        a = self.a_conv1(inputs)
        a = self.a_batch1(a)
        a = torch.tanh(a)
        # layer 2
        a = self.a_conv2(a)
        a = self.a_batch2(a)
        a = torch.tanh(a)

        return a

class attention_block(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = torch.nn.Conv2d(in_channels, 1, kernel_size=1, padding=0)

    def forward(self, input):
        mask = self.attention(input)
        mask = torch.sigmoid(mask)
        B, _, H, W = input.shape
        norm = 2 * torch.norm(mask, p=1, dim=(1, 2, 3))
        norm = norm.reshape(B, 1, 1, 1)
        mask = torch.div(mask * H * W, norm)
        return mask

class appearance_model(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, out_channels):
        # 1st app_block
        super().__init__()
        self.a_block1 = appearnce_block(in_channels, out_channels, kernel_size)
        self.a_dropout2 = torch.nn.Dropout2d(0.5)
        self.a_avg2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.a_mask3 = attention_block(out_channels)
        self.a_block4 = appearnce_block(out_channels, out_channels * 2, kernel_size)
        self.a_mask6 = attention_block(out_channels * 2)

    def forward(self, inputs):
        a = self.a_block1(inputs)
        mask1 = self.a_mask3(a)
        a = self.a_dropout2(a)
        a = self.a_avg2(a)
        a = self.a_block4(a)
        mask2 = self.a_mask6(a)
        return mask1, mask2

class motion_block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, model):
        super().__init__()
        self.model = model

        if self.model.find("TS") is not -1:
            self.m_tsm1 = TSM_block(in_channels,out_channels,kernel_size)
            self.m_tsm2 = TSM_block(out_channels, out_channels, kernel_size)
        else:
            self.m_conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
            self.m_batch1 = torch.nn.BatchNorm2d(out_channels)
            self.m_conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
            self.m_batch2 = torch.nn.BatchNorm2d(out_channels)
        self.m_drop3 = torch.nn.Dropout2d(0.5)
        self.m_avg3 = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, inputs, mask):
        if self.model.find("TS") is not -1:
            m = self.m_tsm1(inputs)
            m = torch.tanh(m)
            m = self.m_tsm2(m)
        else:
            m = self.m_conv1(inputs)
            m = self.m_batch1(m)
            m = torch.tanh(m)

            m = self.m_conv2(m)
            m = self.m_batch2(m)

        m = torch.mul(m, mask)
        m = torch.tanh(m)

        m = self.m_drop3(m)
        m = self.m_avg3(m)
        return m



class motion_model(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, model):
        super().__init__()
        self.m_block1 = motion_block(in_channels, out_channels, kernel_size, model)
        self.m_block2 = motion_block(out_channels, out_channels * 2, kernel_size, model)

    def forward(self, inputs, mask1, mask2):
        m = self.m_block1(inputs, mask1)
        m = self.m_block2(m, mask2)
        return m

class fc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.f_drop1 = torch.nn.Dropout2d(0.5)
        self.f_linear2 = torch.nn.Linear(64 * 9 * 9, 128)
        self.f_linear3 = torch.nn.Linear(128, 1, bias=True)

    def forward(self, input):
        f = torch.flatten(input, start_dim=1)
        f = self.f_drop1(f)
        f = torch.tanh(self.f_linear2(f))
        f = self.f_linear3(f)
        return f

class model(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, model="CAN"):
        super().__init__()
        self.model = model

        self.appearance_model = appearance_model(in_channels=in_channels, out_channels=out_channels,
                                                 kernel_size=kernel_size)
        self.motion_model = motion_model(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, model=model)

        self.fc1 = fc()
        if self.model.find("MT") is not -1:
            self.fc2 = fc()

    def forward(self, norm_input, motion_input):
        mask1, mask2 = self.appearance_model(norm_input)
        out = self.motion_model(motion_input, mask1, mask2)
        out1 = self.fc1(out)
        if self.model.find("MT") is not -1:
            out2 = self.fc2(out)
            return out1, out2
        else:
            return out1
