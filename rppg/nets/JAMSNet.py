import torch
import torchvision.transforms.functional as TF

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class JAMSNet(torch.nn.Module):
    def __init__(self):
        super(JAMSNet, self).__init__()
        self.cnn = torch.nn.Conv1d(1,1,1)
        self.f_1 = Feature_extraction()
        self.f_2 = Feature_extraction()
        self.f_3 = Feature_extraction()
        self.lwaf = LWAF()
        self.rPPG_extraction_net = rPPG_extraction_network()

    def forward(self,x):

        x = torch.permute(x, (0,2,1,3,4))
        x = self.gaussian_pyramid(x)
        x[0] = self.f_1(x[0])
        x[1] = self.f_2(x[1])
        x[2] = self.f_3(x[2])
        x = self.lwaf(x)
        x = self.rPPG_extraction_net(x)
        return x

    def gaussian_pyramid(self, video, num_levels=3):
        pyramid = []
        batch, length, channel, h, w = video.shape

        pyramid.append(video)

        for _ in range(num_levels -1):
            h = h//2
            w = w//2
            pyramid.append(torch.zeros(batch,length,channel, h, w).to(device))

        for i, frame in enumerate(video):
            for j in range(num_levels - 1):
                frame = TF.gaussian_blur(frame, kernel_size=3)
                frame = torch.nn.functional.interpolate(frame, scale_factor=0.5, mode='bilinear', align_corners=False)
                pyramid[j+1][i] = frame
        for i in range(len(pyramid)):
            pyramid[i] = pyramid[i]/255.
        return pyramid

class rPPG_extraction_network(torch.nn.Module):
    def __init__(self):
        super(rPPG_extraction_network, self).__init__()
        self.conv_1 = torch.nn.Conv3d(32, 64, kernel_size=(1, 5, 5), stride=(1, 1, 1), padding=0)
        self.bn_1 = torch.nn.BatchNorm3d(64)
        self.block_1 = rPPG_extraction_block(pool_kernel=[1,2,2],pool_stride=[1,2,2],pool_padding=[0,0,0])
        self.block_2 = rPPG_extraction_block(pool_kernel=[1,2,2],pool_stride=[1,2,2],pool_padding=[0,0,0])
        self.block_3 = rPPG_extraction_block(pool_kernel=[1,2,2],pool_stride=[1,2,2],pool_padding=[0,0,0])
        self.block_4 = rPPG_extraction_block(pool_kernel=[1,2,2],pool_stride=[1,2,2],pool_padding=[0,0,0])
        self.adapt_avg_pool3d = torch.nn.AdaptiveAvgPool3d(output_size=(150, 1, 1))
        self.conv_2 = torch.nn.Conv3d(64, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=0)

    def forward(self,x):
        x = x.unsqueeze(0)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = torch.nn.functional.elu(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.adapt_avg_pool3d(x)
        x = self.conv_2(x)
        return x.view(-1,150)

class  rPPG_extraction_block(torch.nn.Module):
    def __init__(self,pool_kernel, pool_stride, pool_padding):
        super(rPPG_extraction_block, self).__init__()
        self.max_pool_1 = torch.nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_stride,padding=pool_padding)
        self.conv_2 = torch.nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        self.ctja = CTJA()
        self.bn_2 = torch.nn.BatchNorm3d(64)
        self.conv_3 = torch.nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        self.stja = STJA()
        self.bn_3 = torch.nn.BatchNorm3d(64)
    def forward(self,x):
        x = self.max_pool_1(x)
        x = self.conv_2(x)
        x = self.ctja(x)
        x = self.bn_2(x)
        x = torch.nn.functional.elu(x)
        x = self.conv_3(x)
        x = self.stja(x)
        x = self.bn_3(x)
        x = torch.nn.functional.elu(x)
        return x


class Feature_extraction(torch.nn.Module):
    def __init__(self):
        super(Feature_extraction, self).__init__()
        self.conv_1 = torch.nn.Conv2d(3,32,(5,5),(1,1),0)
        self.bn_1 = torch.nn.BatchNorm2d(32)
        self.adaptive_avg_pool = torch.nn.AdaptiveAvgPool2d((48,32))
        self.conv_2 = torch.nn.Conv2d(32,32,(3,3),(1,1),(1,1))
        self.bn_2 = torch.nn.BatchNorm2d(32)
        self.conv_3 = torch.nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1))
        self.bn_3 = torch.nn.BatchNorm2d(32)
        self.STJA = STJA2D()

    def forward(self,x):
        _,_,C, H,W = x.shape
        x = torch.reshape(x, (-1,C, H, W))
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = torch.nn.functional.elu(x)
        x = self.adaptive_avg_pool(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = torch.nn.functional.elu(x)
        x = self.conv_3(x)
        x = self.STJA(x)
        x = self.bn_3(x)
        x = torch.nn.functional.elu(x)
        return x

class LWAF(torch.nn.Module): # Layer-wise Attention Fusion
    def __init__(self):
        super(LWAF, self).__init__()
        self.conv = torch.nn.Conv2d(1,1,1,1,0)
        self.adaptive_avg_pool = torch.nn.AdaptiveAvgPool2d((48,32))
    def forward(self,x):
        L = torch.zeros((1, 1, len(x))).to(device)
        L[:,:,0] = torch.mean(x[0])
        L[:,:,1] = torch.mean(x[1])
        L[:,:,2] = torch.mean(x[2])
        L = self.conv(L)
        L = L.squeeze(0)
        L = torch.softmax(L,1)
        x = x[0] * L[:,0] +  x[1] * L[:,1] +  x[2] * L[:,2]
        x = self.adaptive_avg_pool(x)
        return x

class CTJA(torch.nn.Module):
    def __init__(self):
        super(CTJA, self).__init__()
        self.conv_D13 = torch.nn.Conv3d(1, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1))
        self.conv_D23 = torch.nn.Conv3d(1, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 2, 2), dilation=(1, 2, 2))
        self.conv_D43 = torch.nn.Conv3d(1, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 4, 4), dilation=(1, 4, 4))
        self.bn_1 = torch.nn.BatchNorm3d(3)
        self.conv_DW = torch.nn.Conv3d(3, 3, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1, groups=3)
        self.conv_PW = torch.nn.Conv3d(3, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.bn_2 = torch.nn.BatchNorm3d(1)
    def forward(self,x):
        batch, channel, length, h,w = x.shape
        x_d = torch.mean(x,(3,4),True)
        x_d = x_d.permute(0,4,3,1,2)
        x_d = torch.stack([self.conv_D13(x_d),self.conv_D23(x_d),self.conv_D43(x_d)],0)
        x_d = x_d.squeeze(2).permute(1,0,2,3,4)
        x_d = self.bn_1(x_d)
        x_d = torch.nn.functional.elu(x_d)
        x_d = self.conv_DW(x_d)
        x_d = self.conv_PW(x_d)
        x_d = self.bn_2(x_d)
        x_d = torch.nn.functional.elu(x_d)
        x_d = torch.sigmoid(x_d)
        x_d = x_d.view(1,1,1,length*channel,1)
        x_d = x_d.permute(0,3,4,1,2)
        x_d = x_d.repeat(1,1,1,1,h * w)
        x = x.view(1, channel, length, 1, h*w)
        x = x.view(1, channel*length, 1, 1, h*w)
        x = x_d * x
        x = x.view(1, channel, length, 1, h* w)
        x = x.view(1,channel, length, h, w)
        return x

class STJA(torch.nn.Module):
    def __init__(self):
        super(STJA, self).__init__()
        self.temporal = STJA_Temporal()
        self.spatial = STJA_Spatial()

    def forward(self,x):
        batch, channel, length, h, w = x.shape
        x_t = self.temporal(x)
        x_s = self.spatial(x)
        x_t = x_t.repeat(1,1,1,h,w)
        x_st = x_s * x_t
        x_st = x_st.repeat(1,channel,1,1,1)
        return x * x_st

class STJA_Temporal(torch.nn.Module):
    def __init__(self):
        super(STJA_Temporal, self).__init__()
        self.conv = torch.nn.Conv3d(1,1,(1,1,1),1,0)
        self.bn = torch.nn.BatchNorm3d(1)

    def forward(self,x):
        x = torch.mean(x,(1,3,4),True)
        x = self.conv(x)
        x = self.bn(x)
        x = torch.nn.functional.elu(x)
        return torch.sigmoid(x)

class STJA_Spatial(torch.nn.Module):
    def __init__(self):
        super(STJA_Spatial, self).__init__()
        self.conv = torch.nn.Conv3d(1,1,(3,3,3),1,1)
        self.bn = torch.nn.BatchNorm3d(1)

    def forward(self,x):
        x = torch.mean(x,1,True)
        x = self.conv(x)
        x = self.bn(x)
        x = torch.nn.functional.elu(x)
        return torch.sigmoid(x)

class STJA2D(torch.nn.Module):
    def __init__(self):
        super(STJA2D, self).__init__()
        self.temporal = STJA2D_Temporal()
        self.spatial = STJA2D_Spatial()
    def forward(self,x):
        T, C, H, W = x.shape
        x_t = self.temporal(x)
        x_s = self.spatial(x)
        x_t = x_t.repeat(1,1,H,W)
        x_st = x_s * x_t
        x_st = x_st.repeat(1,C,1,1)
        return x * x_st

class STJA2D_Spatial(torch.nn.Module):
    def __init__(self):
        super(STJA2D_Spatial, self).__init__()
        self.conv = torch.nn.Conv2d(1,1,(3,3),stride=(1,1),padding=1)
        self.bn = torch.nn.BatchNorm2d(1)
    def forward(self,x):
        x = torch.mean(x, 1, True)
        x = self.conv(x)
        x = self.bn(x)
        x = torch.nn.functional.elu(x)
        return torch.sigmoid(x)

class STJA2D_Temporal(torch.nn.Module):
    def __init__(self):
        super(STJA2D_Temporal, self).__init__()
        self.conv = torch.nn.Conv2d(1,1,(1,1),(1,1),0)
        self.bn = torch.nn.BatchNorm2d(1)

    def forward(self,x):

        x = torch.mean(x,dim=(1,2,3),keepdim=True) # Dt 1/c*H*W
        x = self.conv(x)
        x = self.bn(x)
        x = torch.nn.functional.elu(x)
        return torch.sigmoid(x)

