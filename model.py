import torch

class appearnce_block(torch.nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size):
        #layer 1
        self.a_conv1 = torch.nn.Conv2d(in_channels,out_channels,kernel_size,stride=1,padding=1)
        self.a_batch1 = torch.nn.BatchNorm2d(self.a_conv1.out_channels)
        # layer 2
        self.a_conv2 = torch.nn.Conv2d(self.a_conv1.out_channels, out_channels, kernel_size, stride=1, padding=1)
        self.a_batch2 = torch.nn.BatchNorm2d(self.a_conv2.out_channels)

    def forward(self,inputs):
        #layer 1
        a = self.a_conv1(inputs)
        a = self.a_batch1(a)
        a = torch.tanh(a)
        #layer 2
        a = self.a_conv2(a)
        a = self.a_batch2(a)
        attention = torch.tanh(a)

        return a, attention
class attention_block(torch.nn.Module):
    def __init__(self,in_channels):
        self.attention = torch.nn.Conv2d(in_channels,1,kernel_size=1,padding=0)
    def forward(self, input):
        mask = self.attention(input)
        mask = torch.sigmoid(mask)
        B, _, H, W = mask.shape
        norm = 2 * torch.norm(mask, p=1, dim=(1,2,3))
        norm = norm.reshape(B,1,1,1)
        mask = torch.div(mask*H*W,norm)
        return mask
class appearance_model(torch.nn.Module):
    def __init(self,in_channels,kernel_size,out_channels):
        #1st app_block
        self.a_block1 = appearnce_block(in_channels,out_channels,kernel_size)
        self.a_dropout2 = torch.nn.Dropout2d(0.5)
        self.a_avg2 = torch.nn.AvgPool2d(kernel_size=2,stride=2)
        self.a_mask3 = attention_block(out_channels)
        self.a_block4 = appearnce_block(out_channels,out_channels*2,kernel_size)
        self.a_mask6 = attention_block(out_channels*2)

    def forward(self,inputs):
        a,mask1 = self.a_block1(inputs)
        a = self.a_dropout2(a)
        a = self.a_avg2(a)
        mask1 = self.a_mask3(mask1)
        a,mask2 = self.a_block4(a)
        mask2 = self.a_mask6(mask2)
        return mask1,mask2
class motion_block(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size):
        self.m_conv1 = torch.nn.Conv2d(in_channels,out_channels,kernel_size,padding = 1)
        self.m_batch1 = torch.nn.BatchNorm2d(out_channels)
        self.m_conv2 = torch.nn.Conv2d(out_channels,out_channels,kernel_size,padding=1)
        self.m_batch2 = torch.nn.BatchNorm2d(out_channels)
        self.m_drop3 = torch.nn.Dropout2d(0.5)
        self.m_avg3 = torch.nn.AvgPool2d(kernel_size=2,stride=2,padding=0)
    def forward(self,inputs,mask):
        m = self.m_conv1(inputs)
        m = self.m_batch1(m)
        m = torch.tanh(m)

        m = self.m_conv2(m)
        m = self.m_batch2(m)
        m = torch.mul(m,mask)
        m = torch.tanh(m)
        m = self.m_drop3(m)
        m = self.m_avg3(m)
        return m
class motion_model(torch.nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size):
        self.m_block1 = motion_block(in_channels,out_channels,kernel_size)
        self.m_block2 = motion_block(out_channels,out_channels*2,kernel_size)
    def forward(self,inputs,mask1,mask2):
        m =self.m_block1(inputs,mask1)
        m = self.m_block2(m,mask2)
        return m
class fc(torch.nn.Module):
    def __init__(self):
        self.f_linear1 = torch.nn.Linear(64*9*9,128)
        self.f_linear2 = torch.nn.Linear(128,1,bias=True)
    def forward(self,input):
        f = torch.flatten(input)
        f = self.f_linear1(f)
        f = self.f_linear2(f)
        return f
class DeepPhys(torch.nn.Module):
    def __init__(self,in_chnnels,out_channels,kernel_size):
        self.appearance_model = appearance_model(in_chnnels,out_channels,kernel_size)
        self.motion_model = motion_model(in_chnnels,out_channels,kernel_size)
        self.fc = fc()
    def forward(self,norm_input,motion_input):
        mask1,mask2 = self.appearance_model(norm_input)
        out = self.motion_model(motion_input,mask1,mask2)
        out = self.fc(out)
        return out
