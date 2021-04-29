import torch
from functional import Learner
from torch import nn

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
        self.config = [
            ('conv2d',[out_channels,in_channels,kernel_size,kernel_size,1,1]),
            ('bn',[out_channels]),
            ('tanh',[True]),
            ('conv2d', [out_channels, out_channels, kernel_size, kernel_size, 1, 1]),
            ('bn', [out_channels]),
            ('tanh', [True]),
        ]
        self.model = Learner(self.config)

    def forward(self, inputs):
        return self.model.forward(inputs)
class attention_block(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.config = [
            ('conv2d',[1,in_channels,1,1,1,0]),
            ('attention',[])
        ]
        self.model = Learner(self.config)

    def forward(self, input):
        return self.model.forward(input)
class appearance_model(torch.nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size):
        # 1st app_block
        super().__init__()
        self.config = [
            ('appearance_block',[appearnce_block(in_channels=in_channels,out_channels=out_channels, kernel_size=kernel_size)]),
            ('attention_block', [attention_block(out_channels)]),
            ('drop',[0.5]),
            ('avg_pool2d',[2,2,0]),
            ('appearance_block',[appearnce_block(in_channels=out_channels,out_channels=out_channels*2,kernel_size=kernel_size)]),
            ('attention_block',[attention_block(out_channels*2)])
        ]
        self.model = Learner(self.config)

    def forward(self, inputs):
        mask = self.model.forward(inputs)
        return mask
class motion_block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.config = [
            ('conv2d',[out_channels,in_channels,kernel_size,kernel_size,1,1]),
            ('bn',[out_channels]),
            ('tanh',[True]),
            ('conv2d',[out_channels,out_channels,kernel_size,kernel_size,1,1]),
            ('bn',[out_channels]),
            ('mul',[True]),
            ('tanh',[True]),
            ('drop',[0.5]),
            ('avg_pool2d',[2,2,0])
        ]
        self.model = Learner(self.config)

    def forward(self, inputs, mask):
        m = self.model.forward([inputs,mask])
        return m
class motion_model(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.config = [
            [('motion_block',[motion_block(in_channels,out_channels,kernel_size)])],
            [('motion_block',[motion_block(out_channels,out_channels*2,kernel_size)])]
        ]
        #self.model = Learner(self.config)
        self.blocks = [Learner(config) for config in self.config]

    def forward(self, inputs, mask1, mask2):
        out = self.blocks[0].forward([inputs,mask1])
        out = self.blocks[1].forward([out, mask2])
        return out

class fc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = [
            ('flatten',[]),
            ('drop',[0.5]),
            ('linear',[128,64*9*9]),
            ('tanh',[True]),
            ('linear',[1,128])
        ]
        self.model = Learner(self.config)

    def forward(self, input):
        return self.model.forward(input)

class model(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, model="CAN"):
        super().__init__()
        self.model = model
        self.config = [
            [('appearance_model',[appearance_model(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size)])],
            [('motion_model', [motion_model(in_channels, out_channels, kernel_size)])],
            [('fc', [fc()])]
        ]

        self.blocks = [Learner(config) for config in self.config]

    def forward(self, norm_input, motion_input):
        mask = self.blocks[0].forward(norm_input)
        out = self.blocks[1].forward([motion_input,mask[0],mask[1]])
        out = self.blocks[2].forward(out)
        return out
