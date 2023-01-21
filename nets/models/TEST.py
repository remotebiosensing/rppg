import torch
import torch.nn as nn
from nets.modules.vit_pytorch.max_vit import MaxViT,MaxVit_GAN_layer,MaxViT_layer,CrissCrossAttention,FeedForward,MBConv
from einops import rearrange
from einops.layers.torch import Rearrange
import math
import torch.nn.functional as F

import numpy as np

class APNET(nn.Module):
    def __init__(self):
        super(APNET, self).__init__()

        self.length = 32
        height,width = (30,30)
        self.main_conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=(4,height), stride=(1,height), padding=(4//2,1))

        # self.main_conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1)
        self.main_conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1)
        self.main_seqmax = MaxViT_layer(layer_depth=2, layer_dim_in=3, layer_dim=32,
                              kernel=(1,8),dilation=(1,8),padding=0,
                              mbconv_expansion_rate=4, mbconv_shrinkage_rate=0.25, w=4, dim_head=32, dropout=0.2,flag=False)

        self.sa_main = SpatialAttention()

        self.adaptive = nn.AdaptiveAvgPool2d((32, 16))

        self.max_vit = MaxViT_layer(layer_depth=2, layer_dim_in=1, layer_dim=32,
                                    kernel=3, dilation=1, padding=1,
                                    mbconv_expansion_rate=4, mbconv_shrinkage_rate=0.25, w=4, dim_head=32, dropout=0.1,
                                    flag=False)

        self.be_conv1d = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=5//2)
        self.out_conv1d = nn.Conv1d(in_channels=32,out_channels=1,kernel_size=1)

        self.sigmoid = nn.Sigmoid()

        self.conv = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=1)
        self.linear = nn.Linear(8,16)

    def forward_backbone(self, x):
        # x :  b c l h w
        b,c,l,h,w = x.shape
        m0 = x.permute(0, 2, 1, 3, 4) # b l c h w
        m1 = torch.reshape(m0,shape=(-1,1,l,c*h*w))
        # m1 = rearrange(m0, 'b l c h w -> (b l) c h w')  # 1
        m2 = self.main_conv1(m1)  # 2
        m3 = self.main_conv2(m2)  # 3
        m4 = rearrange(m3, '(b l) c h w -> b l c h w', l=self.length)  # 4
        m5 = m4.permute(0, 2, 1, 3, 4)  # 5
        m6 = rearrange(m5, 'b c l h w-> b c l (h w)')  # 6
        m7 = self.main_seqmax(m6)  # 7
        m8 = torch.squeeze(self.sa_main(m7))
        m9 = self.linear(m8)# 8
        # m9 = self.adaptive(m8)  # 9
        m10 = rearrange(m9, 'b l (h w) -> b l h w', h=4, w=4)  # 10
        # main_11 = torch.permute(main_10,(0, 2, 1, 3, 4))

        m13 = rearrange(m10, 'b l h w -> b l (h w)')  # 12

        # o1 = self.conv(m13)
        # o1 = self.max_vit(m13)                                             #13
        # o2 = torch.squeeze(o1)  # 14
        o1 = torch.mean(m13, dim=-1)  # 15
        o3 = torch.reshape(o1,shape=(-1,1,32))
        out_att = self.be_conv1d(o3)
        o4 = (1 + self.sigmoid(out_att)) * o3  # 16
        o5 = self.out_conv1d(o4)  # 17
        out = torch.squeeze(o5)

        return out

    def forward(self,x):

        out = [self.forward_backbone(data) for data in x ]

        return out



class TEST2(nn.Module):
    def __init__(self,ver):
        super(TEST2, self).__init__()

        length = 32
        height,width = (128,128)
        self.ver = ver

        self.main_conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1)
        self.main_conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1)
        self.main_seqmax = MaxViT_layer(layer_depth=2, layer_dim_in=3, layer_dim=32,
                              kernel=(1,32),dilation=(1,32),padding=0,
                              mbconv_expansion_rate=4, mbconv_shrinkage_rate=0.25, w=4, dim_head=32, dropout=0.1,flag=False)

        self.ptt_conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1)
        self.ptt_conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1)
        self.ptt_seqmax = MaxViT_layer(layer_depth=2, layer_dim_in=3, layer_dim=32,
                         kernel=(1,8),dilation=(1,32),padding=0,
                         mbconv_expansion_rate=4, mbconv_shrinkage_rate=0.25, w=4, dim_head=32, dropout=0.1,flag=True)

        self.bvp_conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1)
        self.bvp_conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1)
        self.bvp_seqmax = MaxViT_layer(layer_depth=2, layer_dim_in=3, layer_dim=32,
                         kernel=(1,8),dilation=(1,32),padding=0,
                         mbconv_expansion_rate=4, mbconv_shrinkage_rate=0.25, w=4, dim_head=32, dropout=0.1,flag=True)

        self.sa_main = SpatialAttention()
        self.sa_bvp = SpatialAttention()
        self.sa_ptt = SpatialAttention()

        self.adaptive = nn.AdaptiveAvgPool2d((32, 16))

        self.max_vit = MaxViT_layer(layer_depth=2, layer_dim_in=1, layer_dim=32,
                                    kernel=3, dilation=1, padding=1,
                                    mbconv_expansion_rate=4, mbconv_shrinkage_rate=0.25, w=4, dim_head=32, dropout=0.1,
                                    flag=False)

        self.be_conv1d = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding="same")
        self.out_conv1d = nn.Conv1d(in_channels=32,out_channels=1,kernel_size=1)

        self.sigmoid = nn.Sigmoid()

        self.conv = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=1)

    def forward(self,x):
        length = 32

        m0 = torch.permute(x,(0,2,1,3,4))
        m1 = rearrange(m0,'b l c h w -> (b l) c h w')             #1
        m2 = self.main_conv1(m1)                                  #2
        m3 = self.main_conv2(m2)                                 #3
        m4 = rearrange(m3,'(b l) c h w -> b l c h w', l=length)   #4
        m5 = torch.permute(m4, (0, 2, 1, 3, 4))                   #5
        m6 = rearrange(m5,'b c l h w-> b c l (h w)')              #6
        m7 = self.main_seqmax(m6)                                 #7
        m8 = self.sa_main(m7)                                     #8
        m9 = self.adaptive(m8)                                    #9
        m10 = rearrange(m9,'b c l (h w) -> b c l h w', h=4,w=4)    #10
        # main_11 = torch.permute(main_10,(0, 2, 1, 3, 4))

        p0 = torch.permute(x,(0,3,1,2,4))                                #0
        p1 = rearrange(p0,'b h c l w -> (b h) c l w')               #1
        p2 = self.ptt_conv1(p1)                                     #2
        p3 = self.ptt_conv2(p2)                                     #3
        p4 = rearrange(p3,'(b h) c l w -> b h c l w',h = 128)       #4
        p5 = torch.permute(p4,(0,2,1,3,4))                          #5
        p6 = rearrange(p5,'b c h l w -> b c h (l w)')               #6
        p7 = self.ptt_seqmax(p6)                                    #7
        p8 = self.sa_ptt(p7)                                        #8
        p9 = rearrange(p8,'b c h (l w) -> b c h l w',l=4,w=4)       #9
        p10 = torch.permute(p9,(0,1,3,2,4))                          #10

        b0 = torch.permute(x,(0,4,1,2,3))
        b1 = rearrange(b0,'b w c l h -> (b w) c l h')
        b2 = self.bvp_conv1(b1)
        b3 = self.bvp_conv2(b2)
        b4 = rearrange(b3,'(b w) c l h -> b w c l h',w = 128)
        b5 = torch.permute(b4,(0,2,1,3,4))
        b6 = rearrange(b5, 'b c w l h -> b c w (l h)')
        b7 = self.bvp_seqmax(b6)
        b8 = self.sa_bvp(b7)
        b9 = rearrange(b8,'b c w (l h) -> b c w l h', l =4, h = 4)
        b10 = torch.permute(b9,(0,1,3,4,2))

        if self.ver == 0: # M(W@H)+M
            att = b10 @ p10
            m11 = m10 * F.interpolate(att, scale_factor=(8, 1, 1)) + m10
        elif self.ver == 1:# M(W+H)+M
            att1 = F.interpolate(b10, scale_factor=(1, 1, 1 / 16))  # w
            att2 = F.interpolate(p10, scale_factor=(1,1/16,1)) #H
            att = att1 + att2
            m11 = m10 * F.interpolate(att, scale_factor=(8, 1, 1)) + m10
        elif self.ver == 2:# MW+M
            att = F.interpolate(b10, scale_factor=(1, 1, 1 / 16))  # w
            m11 = m10 * F.interpolate(att, scale_factor=(8, 1, 1)) + m10
        elif self.ver == 3:# MH+M
            att2 = F.interpolate(p10, scale_factor=(1,1/16,1)) #H
            att = att2
            m11 = m10 * F.interpolate(att, scale_factor=(8, 1, 1)) + m10
        elif self.ver == 4: #M
            m11 = m10
        elif self.ver == 5: #WM + HM
            b11 = F.interpolate(b10, scale_factor=(8, 1, 1 / 16))  # W
            p11 = F.interpolate(p10, scale_factor=(8, 1 / 16, 1))  # H
            m11 = m10 *b11 + m10 *p11
        elif self.ver==6: #WM
            b11 = F.interpolate(b10, scale_factor=(8, 1, 1 / 16))  # W
            m11 = m10 * b11
        elif self.ver == 7: #HM
            p11 = F.interpolate(p10, scale_factor=(8, 1 / 16, 1))  # H
            m11 = m10 * p11
        elif self.ver == 8: #H
            p11 = F.interpolate(p10, scale_factor=(8, 1 / 16, 1))  # H
            m11 = p11
        elif self.ver == 9: #W
            b11 = F.interpolate(b10, scale_factor=(8, 1, 1 / 16))  # W
            m11 = b11


        m13 = rearrange(m11, 'b c l w h -> b c l (w h)')                    #12

        o1 = self.conv(m13)
        # o1 = self.max_vit(m13)                                             #13
        o2 = torch.squeeze(o1)                                            #14
        o3 = torch.mean(o2, dim=-1)                                       #15
        out_att = self.be_conv1d(o3)
        o4 = (1 + self.sigmoid(out_att)) * o3                             #16
        o5 = self.out_conv1d(o4)                                          #17
        out = torch.squeeze(o5)

        return out


class TEST(nn.Module):
    def __init__(self):
        super(TEST, self).__init__()

        length = 32
        height,width = (128,128)

        self.main_seq_stem = nn.Sequential(
            Rearrange('b c l h w -> (b l) c h w'),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1),
            Rearrange('(b l) c h w -> b c l (h w)', l=length)
        )
        self.main_seq_max_1 = nn.Sequential(
            MaxViT_layer(layer_depth=2, layer_dim_in=3, layer_dim=32,
                         kernel=(1,32),dilation=(1,32),padding=0,
                         mbconv_expansion_rate=4, mbconv_shrinkage_rate=0.25, w=4, dim_head=32, dropout=0.1,flag=False)
        )

        self.ptt_seq_stem = nn.Sequential(
            Rearrange('b c l h w -> (b h) c l w'),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1),
            Rearrange('(b h) c l w -> b c h (l w)', h=height)
        )
        self.ptt_seq_max_1 = nn.Sequential(
            MaxViT_layer(layer_depth=2, layer_dim_in=3, layer_dim=32,
                         kernel=(1,8),dilation=(1,32),padding=0,
                         mbconv_expansion_rate=4, mbconv_shrinkage_rate=0.25, w=4, dim_head=32, dropout=0.1,flag=True))

        self.bvp_seq_stem = nn.Sequential(
            Rearrange('b c l h w -> (b w) c l h'),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1),
            Rearrange('(b w) c l h -> b c w (l h)', w=width)
        )
        self.bvp_seq_max_1 = nn.Sequential(
            MaxViT_layer(layer_depth=2, layer_dim_in=3, layer_dim=32,
                         kernel=(1,8),dilation=(1,32),padding=0,
                         mbconv_expansion_rate=4, mbconv_shrinkage_rate=0.25, w=4, dim_head=32, dropout=0.1,flag=True))
        self.max_vit = MaxViT_layer(layer_depth=2, layer_dim_in=1, layer_dim=32,
                         kernel=3, dilation=1, padding=1,
                         mbconv_expansion_rate=4, mbconv_shrinkage_rate=0.25, w=4, dim_head=32, dropout=0.1,flag=False)
        self.adaptation = nn.AdaptiveAvgPool2d((4,2))

        self.sa_main = SpatialAttention()
        self.sa_bvp = SpatialAttention()
        self.sa_ptt = SpatialAttention()

        self.adaptive = nn.AdaptiveAvgPool2d((32,16))
        self.be_conv1d = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding="same")
        self.out_conv1d = nn.Conv1d(in_channels=32,out_channels=1,kernel_size=1)

        self.sigmoid = nn.Sigmoid()



    def forward(self,x):
        main_1 = self.main_seq_stem(x)
        main_2 = self.main_seq_max_1(main_1)
        #ver1
        main_3 = self.sa_main(main_2)
        #ver2
        # main_att = self.sa_main(main)
        # main = main_att*main + main
        main_4 = self.adaptive(main_3)
        main_5 = rearrange(main_4,'b c l (w h) -> b c l w h',w = 4, h = 4)
        # main = self.main_seq_max_2(main)

        bvp_1 = self.bvp_seq_stem(x)
        bvp_2 = self.bvp_seq_max_1(bvp_1)
        #ver1
        bvp_3 = self.sa_bvp(bvp_2)
        #ver2
        # bvp_att = self.sa_bvp(bvp)
        # bvp = bvp_att*bvp + bvp
        bvp_4 = rearrange(bvp_3, 'b c w (l h) -> b c l w h', l=4, h=4)


        ptt_1 = self.ptt_seq_stem(x)
        ptt_2 = self.ptt_seq_max_1(ptt_1)
        #ver1
        ptt_3 = self.sa_bvp(ptt_2)
        #ver2
        # ptt_att = self.sa_bvp(ptt)
        # ptt = ptt_att*ptt + ptt
        ptt_4 = rearrange(ptt_3, 'b c h (l w) -> b c l w h', l=4, w=4)

        # att = ptt_4@bvp_4
        att = F.interpolate(ptt_4,scale_factor=(1,1,1/16))
        main_6 = main_5 * F.interpolate(att,scale_factor=(8,1,1)) + main_5

        main_7 = rearrange(main_6,'b c l w h -> b c l (w h)')
        out_1 = self.max_vit(main_7)

        out_2 = torch.squeeze(out_1)
        out_3 = torch.mean(out_2,dim = -1)

        out_att = self.be_conv1d(out_3)
        out_4 = (1 + self.sigmoid(out_att)) * out_3
        out_5 = self.out_conv1d(out_4)
        out = torch.squeeze(out_5)
        # out = self.linear(out)
        return out
            # ,[main_1,main_2,main_3,main_4,main_5,main_6,main_7],\
            #    [bvp_1,bvp_2,bvp_3,bvp_4],[ptt_1,ptt_2,ptt_3,ptt_4],[att,out_att],\
            #    [out_1,out_2,out_3,out_4,out_5]


class SpatialAttention(nn.Module):
    def __init__(self, kernel=3):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel, padding=kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)

        return self.sigmoid(x)