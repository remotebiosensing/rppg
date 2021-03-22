import torch

class appearnce_block(torch.nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size):
        #layer 1
        self.a_conv1 = torch.nn.Conv2d(in_channels,out_channels,kernel_size,stride=1,paddinf=1)
        self.a_batch1 = torch.nn.BatchNorm2d(self.a_conv1.out_channels)
        # layer 2
        self.a_conv2 = torch.nn.Conv2d(self.a_conv1.out_channels, out_channels, kernel_size, stride=1, paddinf=1)
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
        self.a_mask3 = attention_block(self.a_block1.a_conv2.out_channels)
        self.a_block4 = appearnce_block(self.a_block1.a_conv2.out_channels,out_channels*2,kernel_size)
        self.a_dropout5 = torch.nn.Dropout2d(0.5)
        self.a_avg5 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.a_mask6 = attention_block(self.a_block4.a_conv2.out_channels)

    def forward(self,inputs):
        a,mask1 = self.a_block1(inputs)
        a = self.a_dropout2(a)
        a = self.a_avg2(a)
        mask1 = self.a_mask3(mask1)
        a,mask2 = self.a_block4(a)
        a = self.a_dropout5(a)
        a = self.a_avg5(a)
        mask2 = self.a_mask6(mask2)
        return a, mask1,mask2

