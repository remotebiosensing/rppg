from functools import partial

import torch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
import math

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# helper classes

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x

# class BatchNormResidual(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.norm = nn.BatchNorm2d(dim)
#         self.fn = fn
#
#     def forward(self, x):
#         return self.fn(self.norm(x)) + x

class BatchNormRes(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):

        return self.norm(x) + x
class BatchNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)
        self.fn = fn

    def forward(self, x):
        return self.norm(self.fn(x)) + x

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# MBConv

class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, hidden_dim, bias = False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias = False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1')
        )

    def forward(self, x):
        return x * self.gate(x)


class MBConvResidual(nn.Module):
    def __init__(self, fn, dropout = 0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x

class Dropsample(nn.Module):
    def __init__(self, prob = 0):
        super().__init__()
        self.prob = prob
  
    def forward(self, x):
        device = x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1), device = device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)

def MBConv(
    dim_in,
    dim_out,
    kernel,
    dilation,
    padding,
    *,
    downsample,
    expansion_rate = 4,
    shrinkage_rate = 0.25,
    dropout = 0.
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = nn.Sequential(
        nn.Conv2d(dim_in, dim_out, 1),
        nn.BatchNorm2d(dim_out),
        nn.SiLU(),
        # nn.Conv2d(dim_out, dim_out, 3, stride = stride, padding = 1, groups = dim_out),
        # nn.Conv2d(dim_out, dim_out, kernel_size=kernel, stride=stride, padding=padding, dilation=dilation, groups=dim_out),
        nn.Conv2d(dim_out, dim_out, kernel_size=kernel, stride=stride, padding=padding, dilation=dilation,
                  groups=dim_out),
        SqueezeExcitation(dim_out, shrinkage_rate = shrinkage_rate),
        nn.Conv2d(dim_out, dim_out, 1),
        nn.BatchNorm2d(dim_out)
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout = dropout)

    return net

# attention related classes

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        dropout = 0.,
        window_size = 7
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias = False),
            nn.Dropout(dropout)
        )

        # relative positional bias

        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

        pos = torch.arange(window_size)
        # grid = torch.stack(torch.meshgrid(pos, pos, indexing = 'ij'))
        grid = torch.stack(torch.meshgrid(pos,pos))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim = -1)

        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)

    def forward(self, x):
        batch, height, width, window_height, window_width, _, device, h = *x.shape, x.device, self.heads

        # flatten

        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')

        # project for queries, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # split heads

        q, k, v = map(lambda t: rearrange(t, 'b n (h d ) -> b h n d', h = h), (q, k, v))

        # scale

        q = q * self.scale

        # sim

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add positional bias

        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        # attention

        attn = self.attend(sim)

        # aggregate

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads

        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1 = window_height, w2 = window_width)

        # combine heads out

        out = self.to_out(out)
        return rearrange(out, '(b x y) ... -> b x y ...', x = height, y = width)

class MaxViT(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        dim,
        depth,
        dim_head = 32,
        dim_conv_stem = None,
        window_size = 7,
        mbconv_expansion_rate = 4,
        mbconv_shrinkage_rate = 0.25,
        dropout = 0.1,
        channels = 3
    ):
        super().__init__()
        assert isinstance(depth, tuple), 'depth needs to be tuple if integers indicating number of transformer blocks at that stage'

        # convolutional stem

        dim_conv_stem = default(dim_conv_stem, dim)

        self.conv_stem = nn.Sequential(
            nn.Conv2d(channels, dim_conv_stem, 3, stride = 2, padding = 1),
            nn.Conv2d(dim_conv_stem, dim_conv_stem, 3, padding = 1)
        )

        # variables

        num_stages = len(depth)

        dims = tuple(map(lambda i: (2 ** i) * dim, range(num_stages)))
        dims = (dim_conv_stem, *dims)
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        self.layers = nn.ModuleList([])

        # shorthand for window size for efficient block - grid like attention

        w = window_size

        # iterate through stages

        for ind, ((layer_dim_in, layer_dim), layer_depth) in enumerate(zip(dim_pairs, depth)):
            for stage_ind in range(layer_depth):
                is_first = stage_ind == 0
                stage_dim_in = layer_dim_in if is_first else layer_dim
                print(stage_dim_in)
                block = nn.Sequential(
                    MBConv(
                        stage_dim_in,
                        layer_dim,
                        downsample = is_first,
                        expansion_rate = mbconv_expansion_rate,
                        shrinkage_rate = mbconv_shrinkage_rate
                    ),
                    Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1 = w, w2 = w),  # block-like attention
                    PreNormResidual(layer_dim, Attention(dim = layer_dim, dim_head = dim_head, dropout = dropout, window_size = w)),
                    PreNormResidual(layer_dim, FeedForward(dim = layer_dim, dropout = dropout)),
                    Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),

                    Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1 = w, w2 = w),  # grid-like attention
                    PreNormResidual(layer_dim, Attention(dim = layer_dim, dim_head = dim_head, dropout = dropout, window_size = w)),
                    PreNormResidual(layer_dim, FeedForward(dim = layer_dim, dropout = dropout)),
                    Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)'),
                )

                self.layers.append(block)

        # mlp head out

        self.mlp_head = nn.Sequential(
            Reduce('b d h w -> b d', 'mean'),
            nn.LayerNorm(dims[-1]),
            nn.Linear(dims[-1], num_classes)
        )

    def forward(self, x):
        x = self.conv_stem(x)

        for stage in self.layers:
            x = stage(x)

        return x#self.mlp_head(x)

class MaxViT_Block(nn.Module):
    def __init__(self, stage_dim_in, layer_dim,kernel,dilation,padding, is_first, mbconv_expansion_rate,mbconv_shrinkage_rate,w,dim_head,dropout):
        super(MaxViT_Block, self).__init__()
        # self.block = nn.Sequential(
        # self.l1 = MBConv(
        #         stage_dim_in,
        #         layer_dim,
        #         kernel,
        #         dilation,
        #         padding,
        #         downsample=is_first,
        #         expansion_rate=mbconv_expansion_rate,
        #         shrinkage_rate=mbconv_shrinkage_rate
        #     )


        # stride = 2 if 0 else 1

        self.l0 = nn.Sequential(
            nn.Conv2d(stage_dim_in, layer_dim, 1),
            nn.BatchNorm2d(layer_dim),
            nn.SiLU(),

        )
        self.l1 = nn.Sequential(
            nn.Conv2d(layer_dim, layer_dim, kernel_size=kernel, stride=1, padding='same', dilation=dilation,groups=layer_dim),
            SqueezeExcitation(layer_dim, shrinkage_rate=mbconv_shrinkage_rate),
            nn.Conv2d(layer_dim, layer_dim, 1),
            nn.BatchNorm2d(layer_dim)
        )
        self.l2 = Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1=w, w2=w)  # block-like attention
        self.l3 = PreNormResidual(layer_dim, Attention(dim=layer_dim, dim_head=dim_head, dropout=dropout, window_size=w))
        self.l4 = PreNormResidual(layer_dim, FeedForward(dim=layer_dim, dropout=dropout))
        self.l5 = Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)')

        self.l6 = Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1=w, w2=w)  # grid-like attention
        self.l7 = PreNormResidual(layer_dim, Attention(dim=layer_dim, dim_head=dim_head, dropout=dropout, window_size=w))
        self.l8 = PreNormResidual(layer_dim, FeedForward(dim=layer_dim, dropout=dropout))
        self.l9 = Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)')
        # )
    def forward(self,x):
        x = self.l0(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        return x


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)
class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0,
                                                                                                                 2,
                                                                                                                 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0,
                                                                                                                 2,
                                                                                                                 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize,
                                                                                                     width, height,
                                                                                                     height).permute(
            0, 2, 1, 3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height,
                                                                                 height)
        # print(concate)
        # print(att_H)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2,
                                                                                                             3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2,
                                                                                                             1, 3)
        # print(out_H.size(),out_W.size())
        return self.gamma * (out_H + out_W) + x

class MaxVit_GAN_Block(nn.Module):
    def __init__(self, stage_dim_in, layer_dim, mbconv_expansion_rate,mbconv_shrinkage_rate,w,dim_head,dropout):
        super(MaxVit_GAN_Block, self).__init__()
        self.block = nn.Sequential(
            CrissCrossAttention(stage_dim_in),
            BatchNormResidual(stage_dim_in, CrissCrossAttention(stage_dim_in)),
            Rearrange('b d x y -> b x y d'),
            FeedForward(dim=stage_dim_in, dropout=dropout),
            Rearrange('b x y d -> b d x y'),
            # x=rearrange(x, 'b x y d -> b d x y')
            BatchNormRes(stage_dim_in))

        self.grid = nn.Sequential(
            Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1=w, w2=w),  # grid-like attention
            Attention(dim=stage_dim_in, dim_head=dim_head, dropout=dropout, window_size=w),
            Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)'),
            BatchNormRes(stage_dim_in),

            Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1=w, w2=w),  # grid-like attention
            FeedForward(dim=stage_dim_in, dropout=dropout),
            Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)'),
            BatchNormRes(stage_dim_in)
        )
        self.block_like = nn.Sequential(

            # BatchNormResidual(layer_dim, Attention(dim=layer_dim, dim_head=dim_head, dropout=dropout, window_size=w)),
            # BatchNormResidual(layer_dim, FeedForward(dim=layer_dim, dropout=dropout)),


            Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1=w, w2=w),  # block-like attention
            Attention(dim=stage_dim_in, dim_head=dim_head, dropout=dropout, window_size=w),
            Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),
            BatchNormRes(stage_dim_in),

            Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1=w, w2=w),  # block-like attention
            FeedForward(dim=stage_dim_in, dropout=dropout),
            Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),
            BatchNormRes(stage_dim_in))

        self.Mbconv = nn.Sequential(

            MBConv(
                stage_dim_in,
                layer_dim,
                kernel=3,
                dilation=1,
                padding=1,
                downsample=False,
                expansion_rate=mbconv_expansion_rate,
                shrinkage_rate=mbconv_shrinkage_rate
            ),
            #
        )

    def forward(self,x):
        x = self.block(x)
        x = self.grid(x)
        x = self.block_like(x)
        x = self.Mbconv(x)
        return x



class MaxViT_layer(nn.Module):
    def __init__(self,layer_depth,layer_dim_in, layer_dim,kernel,dilation,padding, mbconv_expansion_rate,mbconv_shrinkage_rate,w,dim_head,dropout,flag):
        super(MaxViT_layer, self).__init__()
        self.layers = nn.ModuleList([])
        for i in range(layer_depth):
            is_first = i == 0
            stage_dim_in = layer_dim_in if is_first else layer_dim
            kernel = kernel if is_first else 3
            dilation = dilation if is_first else 1
            padding = padding if is_first else 1
            if flag == True:
                is_first = i == 0
            else:
                is_first = False
            self.layers.append(MaxViT_Block(stage_dim_in,layer_dim,kernel,dilation,padding,is_first,mbconv_expansion_rate,mbconv_shrinkage_rate,w,dim_head,dropout))



    def forward(self,x):
        for stage in self.layers:
            x = stage(x)
        return x

class MaxVit_GAN_layer(nn.Module):
    def __init__(self,layer_depth, layer_dim_in, layer_dim, mbconv_expansion_rate,mbconv_shrinkage_rate,w,dim_head,dropout,scale_factor):
        super(MaxVit_GAN_layer, self).__init__()
        self.layers = nn.ModuleList([])
        for i in range(layer_depth):
            is_first = i == 0
            stage_dim_in = layer_dim_in if is_first else layer_dim
            self.layers.append(MaxVit_GAN_Block(stage_dim_in,layer_dim,mbconv_expansion_rate,mbconv_shrinkage_rate,w,dim_head,dropout))
        # self.layers.append(nn.Upsample(scale_factor=2))
        self.up = nn.Upsample(scale_factor=scale_factor)

    def forward(self,x):
        for stage in self.layers:
            x = stage(x)
        x = self.up(x)
        return x

