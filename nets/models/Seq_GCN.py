import numpy as np
from torch.autograd import Variable
import torch
from torch import einsum
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange,repeat
from einops.layers.torch import Rearrange
import math
from typing import Optional
from self_attention_cv.common import expand_to_batch
import torch
from torch import nn

class Seq_GCN(nn.Module):
    def __init__(self):
        super(Seq_GCN, self).__init__()
        length, height, width = (32,128,128)
        # self.main_plane_module = PlaneModule(img_dim=(length,height*width),patch_dim=(length//8,height*width))
        # self.bvp_plane_module = PlaneModule(img_dim=(height,length*width),patch_dim=(height//8,length*width))
        # self.ptt_plane_module = PlaneModule(img_dim=(width,length*height),patch_dim=(width//8,length*height))
        # self.main_plane_module = PlaneModule(img_dim=(128,128),patch_dim=(16,16))
        # self.bvp_plane_module = PlaneModule()
        # self.ptt_plane_module = PlaneModule()

        # self.seq = nn.Sequential(
        #     nn.Conv2d(in_channels=in_dim,out_channels=out_dim,kernel_size=(3,3),stride=(2,2),padding=(1,1)),
        #     nn.BatchNorm2d(out_dim),
        #     nn.Conv2d(in_channels=out_dim,out_channels=out_dim,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
        #     nn.BatchNorm2d(out_dim),
        #     nn.ReLU(inplace=True)
        # )
        self.involve_main_conv2d = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(4,4),stride=4)
        self.involve_ptt_conv2d = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(4, 4), stride=4)
        self.involve_bvp_conv2d = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(4, 4), stride=4)
        self.main_conv2d_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(1, 32), stride=(1, 1), dilation=(1, 32)),
            nn.BatchNorm2d(32),
        )
        self.main_conv2d_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.main_conv_block = ConvBlock(32,64)
        self.main_project_vit = ViT(img_dim=(8,8),in_channels=64,patch_dim=(1,1),classification=False)

        self.ptt_conv2d_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(1, 8), stride=(1, 1), dilation=(1, 32)),
            nn.BatchNorm2d(32),
        )
        self.ptt_conv2d_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32,kernel_size=(3,3),stride=(2,2),padding=(1,1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.ptt_conv_block = ConvBlock(32, 64)
        self.ptt_project_vit = ViT(img_dim=(32, 8), in_channels=64, patch_dim=(4, 1), classification=False)

        self.bvp_conv2d = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(1, 8), stride=(1, 1), dilation=(1, 32)),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32,kernel_size=(3,3),stride=(2,2),padding=(1,1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.bvp_conv_block = ConvBlock(32, 64)
        self.bvp_project_vit = ViT(img_dim=(32, 8), in_channels=64, patch_dim=(4, 1), classification=False)

        self.bvp_projection = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(8, 1))

        self.avg_pool_4x1 = nn.AvgPool2d(kernel_size=(1,2))

        self.conv_ptt_1d = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=2,stride=2)
        self.conv_bvp_1d = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=2, stride=2)

        self.main_vit = ViT(img_dim=(8,264),in_channels=64,patch_dim=(2,132),dim=32*32,blocks=2,classification=False)

        self.up_1 = UpBlock(32,16)
        self.up_2 = UpBlock(16, 4)
        self.up_3 = UpBlock(4, 1)

        self.batch_norm = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=True)

        # self.conv_main_2d = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(2,5), stride=2)

        # self.main_conv2d = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(1, 32),stride=(1,1), dilation=(1, 32))
        # self.ptt_conv2d = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(2, 64),stride=(1,1), dilation=(1, 32))
        # self.bvp_conv2d = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(2, 64),stride=(1,1), dilation=(1, 32))

    def forward(self,x):
        batch, channel, length, height, width = x.shape
        # 4/3/32/128/128[ batch, channel, length, height, width]
        # l 하나가 이미지 한장
        main_plane = rearrange(x, 'b c l h w -> (b l) c h w') # 128/3/128/128
        main_plane = self.involve_main_conv2d(main_plane)     # 128/3/32/32
        main_plane = self.batch_norm(main_plane)
        main_plane = self.relu(main_plane)
        main_plane = rearrange(main_plane, '(b l) c h w -> b c l (h w)',l=length)  # 4/3/32/1024
        main_plane = self.main_conv2d_1(main_plane)   # 4/3/32/32
        main_plane = self.main_conv2d_2(main_plane)   # 4/32/16/16

        main_plane = self.main_conv_block(main_plane) # 4/64/8/8 # b c l (h w)
        # main_plane = self.main_project_vit(main_plane) # 4/64/64

        # h 하나가 위에 한줄
        ptt_plane = rearrange(x, 'b c l h w -> (b h) c l w')     # 512/3/32/128
        ptt_plane = self.involve_ptt_conv2d(ptt_plane)          # 512/3/8/32
        ptt_plane = self.batch_norm(ptt_plane)
        ptt_plane = self.relu(ptt_plane)
        ptt_plane = rearrange(ptt_plane, '(b h) c l w -> b c h (l w)',h=height)  # 4/3/128/256
        ptt_plane = self.ptt_conv2d_1(ptt_plane) # 4/32/128/32
        ptt_plane = self.ptt_conv2d_2(ptt_plane) # 4/32/64/16

        ptt_plane = self.ptt_conv_block(ptt_plane) #4/64/32/8 # b c h (l w)
        ptt_plane = rearrange(ptt_plane, 'b c h e -> b c (h e)')
        # ptt_plane = self.conv_ptt_1d(ptt_plane)
        ptt_plane = self.avg_pool_4x1(ptt_plane)


        # ptt_plane = self.ptt_project_vit(ptt_plane) # 4/64/256
        # h 당 bvp 피쳐 : 1차 곱

        # w 하나가 세로 한줄 좌측부터
        bvp_plane = rearrange(x, 'b c l h w -> (b w) c l h') # 512/3/32/128
        bvp_plane = self.involve_bvp_conv2d(bvp_plane)       # 512/3/8/32
        bvp_plane = self.batch_norm(bvp_plane)
        bvp_plane = self.relu(bvp_plane)
        bvp_plane = rearrange(bvp_plane, '(b w) c l h -> b c w (l h)',w = width) # 4/3/128/256
        bvp_plane = self.bvp_conv2d(bvp_plane) # 4/32/64/16

        bvp_plane = self.bvp_conv_block(bvp_plane) # 4/64/32/8 # b c w (l h)
        bvp_plane = rearrange(bvp_plane, 'b c w e -> b c (w e)')
        # bvp_plane = self.conv_bvp_1d(bvp_plane)
        bvp_plane = self.avg_pool_4x1(bvp_plane)
        # bvp_plane = self.bvp_project_vit(bvp_plane)

        # w 당 ptt 피쳐 : 1차 곱

        # main : 4/64/8/8  b c l (h w)
        # ptt  : 4/64/32/8 b c h (l w)
        # bvp  : 4/64/32/8 b c w (l h)


        # main_plane = self.main_plane_module(main_plane)
        # ptt_plane = self.ptt_plane_module(main_plane)
        # bvp_plane = self.bvp_plane_module(main_plane)



        # main_plane = rearrange(x, 'b c l h w -> (b l) c h w')
        # main_plane = self.main_plane_module(main_plane)
        # main_plane = rearrange(main_plane, '(b l) c h w -> b c l h w', b = batch, l= length,c=64,h=32,w=32)
        #
        # ptt_plane = rearrange(x, 'b c l h w -> (b w) c h l', b=batch, c=channel, l=length, h=height, w=width)
        # ptt_plane = self.ptt_plane_module(ptt_plane)
        # ptt_plane = rearrange(ptt_plane, '(b w) c h l -> b c l h w', b=batch, c=64, l=8, h=32, w=width)
        #
        # bvp_plane = rearrange(x, 'b c l h w -> (b h) c w l', b = batch, c = channel, l = length, h = height, w =  width)
        # bvp_plane = self.bvp_plane_module(bvp_plane)
        # bvp_plane = rearrange(bvp_plane, '(b h) c w l -> b c l h w', b=batch, c=64, l=8, h=height, w=32)

        out = []
        batch, channel, length, e = main_plane.shape
        for i in range(length):
            out.append(torch.unsqueeze(torch.cat([main_plane[:,:,i,:],ptt_plane,bvp_plane],dim=2),dim=2))

        out = torch.cat(out,dim=2)
        out = self.main_vit(out)
        out = rearrange(out, 'b xy (p c) -> b c xy p',c =32,xy=8)
        # 'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
        # out = self.conv_main_2d(out)
        out = self.up_1(out)
        out = self.up_2(out)
        out = self.up_3(out)
        out = torch.squeeze(out)
        return out

class PlaneModule(nn.Module):
    def __init__(self,img_dim=(128,32),patch_dim=(16,8)):
        super(PlaneModule, self).__init__()

        self.img_dim = img_dim
        self.patch_dim = patch_dim

        self.vit_1 = ViT(img_dim=self.img_dim,in_channels=3, blocks=1, patch_dim=self.patch_dim,classification=False)
        # 16 * 16 * 3 = 768
        self.vit_2 = ViT(img_dim=(self.img_dim[0]//2,self.img_dim[1]//2),in_channels=32, blocks=1, patch_dim=(self.patch_dim[0]//2,self.patch_dim[1]//2),classification=False)


        self.d1 = ConvBlock(3, 32)
        self.d2 = ConvBlock(32, 64)


    def forward(self,x):
        plane = self.vit_1(x)
        plane = rearrange(plane, 'b (x y) (patch_x patch_y c) -> b c (patch_x x) (patch_y y)', patch_x = self.patch_dim[0], patch_y = self.patch_dim[1], c = 3,x = self.img_dim[0]//self.patch_dim[0], y = self.img_dim[1]//self.patch_dim[1])
        plane = self.d1(plane)

        plane = self.vit_2(plane)
        plane = rearrange(plane, 'b (x y) (patch_x patch_y c) -> b c (patch_x x) (patch_y y)', patch_x = self.patch_dim[0]//2, patch_y = self.patch_dim[1]//2, c = 32,x = self.img_dim[0]//self.patch_dim[0], y = self.img_dim[1]//self.patch_dim[1])
        plane = self.d2(plane)


        return plane


class TransformerEncoder(nn.Module):
    def __init__(self, dim, blocks=6, heads=8, dim_head=None, dim_linear_block=1024, dropout=0, prenorm=False):
        super().__init__()
        self.block_list = [TransformerBlock(dim, heads, dim_head,
                                            dim_linear_block, dropout, prenorm=prenorm) for _ in range(blocks)]
        self.layers = nn.ModuleList(self.block_list)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
class TransformerBlock(nn.Module):
    """
    Vanilla transformer block from the original paper "Attention is all you need"
    Detailed analysis: https://theaisummer.com/transformer/
    """

    def __init__(self, dim, heads=8, dim_head=None,
                 dim_linear_block=1024, dropout=0.1, activation=nn.GELU,
                 mhsa=None, prenorm=False):
        """
        Args:
            dim: token's vector length
            heads: number of heads
            dim_head: if none dim/heads is used
            dim_linear_block: the inner projection dim
            dropout: probability of droppping values
            mhsa: if provided you can change the vanilla self-attention block
            prenorm: if the layer norm will be applied before the mhsa or after
        """
        super().__init__()
        self.mhsa = mhsa if mhsa is not None else MultiHeadSelfAttention(dim=dim, heads=heads, dim_head=dim_head)
        self.prenorm = prenorm
        self.drop = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)

        self.linear = nn.Sequential(
            nn.Linear(dim, dim_linear_block),
            activation(),  # nn.ReLU or nn.GELU
            nn.Dropout(dropout),
            nn.Linear(dim_linear_block, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        if self.prenorm:
            y = self.drop(self.mhsa(self.norm_1(x), mask)) + x
            out = self.linear(self.norm_2(y)) + y
        else:
            y = self.norm_1(self.drop(self.mhsa(x, mask)) + x)
            out = self.norm_2(self.linear(y) + y)
        return out
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None):
        """
        Implementation of multi-head attention layer of the original transformer model.
        einsum and einops.rearrange is used whenever possible
        Args:
            dim: token's dimension, i.e. word embedding vector size
            heads: the number of distinct representations to learn
            dim_head: the dim of the head. In general dim_head<dim.
            However, it may not necessary be (dim/heads)
        """
        super().__init__()
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
        _dim = self.dim_head * heads
        self.heads = heads
        self.to_qvk = nn.Linear(dim, _dim * 3, bias=False)
        # self.to_qv = nn.Linear(dim, _dim * 2, bias=False)
        self.W_0 = nn.Linear(_dim, dim, bias=False)
        self.scale_factor = self.dim_head ** -0.5

    def forward(self, x, mask=None):
        assert x.dim() == 3
        qvk = self.to_qvk(x)  # [batch, tokens, dim*3*heads ]

        # decomposition to q,v,k and cast to tuple
        # the resulted shape before casting to tuple will be: [3, batch, heads, tokens, dim_head]
        q,k, v = tuple(rearrange(qvk, 'b t (d k h ) -> k b h t d ', k=3, h=self.heads))
        # k = rearrange(mask, 'b t (d h ) -> b h t d ',h=self.heads)
        out = compute_mhsa(q, k, v, mask=None, scale_factor=self.scale_factor)

        # re-compose: merge heads with dim_head
        out = rearrange(out, "b h t d -> b t (h d)")
        # Apply final linear transformation layer
        return self.W_0(out)
def compute_mhsa(q, k, v, scale_factor=1, mask=None):
    # resulted shape will be: [batch, heads, tokens, tokens]
    scaled_dot_prod = torch.einsum('... i d , ... j d -> ... i j', q, k) * scale_factor

    if mask is not None:
        assert mask.shape == scaled_dot_prod.shape[2:]
        scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)

    attention = torch.softmax(scaled_dot_prod, dim=-1)
    # calc result per head
    return torch.einsum('... i j , ... j d -> ... i d', attention, v)
class ViT(nn.Module):
    def __init__(self, *,
                 img_dim=(256,256),
                 in_channels=3,
                 patch_dim=(16,16),
                 num_classes=10,
                 dim=None,
                 blocks=6,
                 heads=4,
                 dim_linear_block=1024,
                 dim_head=None,
                 dropout=0, transformer=None, classification=True):
        """
        Minimal re-implementation of ViT
        Args:
            img_dim: the spatial image size
            in_channels: number of img channels
            patch_dim: desired patch dim
            num_classes: classification task classes
            dim: the linear layer's dim to project the patches for MHSA
            blocks: number of transformer blocks
            heads: number of heads
            dim_linear_block: inner dim of the transformer linear block
            dim_head: dim head in case you want to define it. defaults to dim/heads
            dropout: for pos emb and transformer
            transformer: in case you want to provide another transformer implementation
            classification: creates an extra CLS token that we will index in the final classification layer
        """
        super().__init__()
        assert img_dim[0] % patch_dim[0] == 0, f'patch size h {patch_dim[0]} not divisible by img dim {img_dim[0]}'
        assert img_dim[1] % patch_dim[1] == 0, f'patch size w {patch_dim[1]} not divisible by img dim {img_dim[1]}'
        # self.p = patch_dim
        self.p_h = patch_dim[0]
        self.p_w = patch_dim[1]
        self.classification = classification
        # tokens = number of patches
        # tokens = (img_dim // patch_dim) ** 2
        tokens = (img_dim[0]//patch_dim[0]) *(img_dim[1]//patch_dim[1])
        #self.token_dim = in_channels * ( patch_dim ** 2)
        self.token_dim = in_channels * (patch_dim[0] * patch_dim[1])
        # self.token_dim = self.p_h*self.p_w*in_channels
        if dim is None:
            self.dim = self.token_dim
        else:
            self.dim = dim
        self.dim_head = (int(self.dim / heads)) if dim_head is None else dim_head

        # Projection and pos embeddings
        self.project_patches = nn.Linear(self.token_dim, self.dim)

        self.emb_dropout = nn.Dropout(dropout)

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.pos_emb1D = nn.Parameter(torch.randn(tokens + 1, self.dim))
        self.mlp_head = nn.Linear(self.dim, num_classes)

        if transformer is None:
            self.transformer = TransformerEncoder(self.dim, blocks=blocks, heads=heads,
                                                  dim_head=self.dim_head,
                                                  dim_linear_block=dim_linear_block,
                                                  dropout=dropout)
        else:
            self.transformer = transformer

    def forward(self, img, mask=None):
        # Create patches
        # from [batch, channels, h, w] to [batch, tokens , N], N=p*p*c , tokens = h/p *w/p
        img_patches = rearrange(img,
                                'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
                                patch_x=self.p_h, patch_y=self.p_w)

        batch_size, tokens, _ = img_patches.shape

        # project patches with linear layer + add pos emb
        img_patches = self.project_patches(img_patches)

        img_patches = torch.cat((expand_to_batch(self.cls_token, desired_size=batch_size), img_patches), dim=1)

        # add pos. embeddings. + dropout
        # indexing with the current batch's token length to support variable sequences
        img_patches = img_patches + self.pos_emb1D[:tokens + 1, :]
        patch_embeddings = self.emb_dropout(img_patches)

        # feed patch_embeddings and output of transformer. shape: [batch, tokens, dim]
        y = self.transformer(patch_embeddings, mask)

        # we index only the cls token for classification. nlp tricks :P
        return self.mlp_head(y[:, 0, :]) if self.classification else y[:, 1:, :]

class ConvBlock(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(ConvBlock, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=in_dim,out_channels=out_dim,kernel_size=(3,3),stride=(2,2),padding=(1,1)),
            nn.BatchNorm2d(out_dim),
            nn.Conv2d(in_channels=out_dim,out_channels=out_dim,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
        # (25 - 3)/2 +1

    def forward(self,x):
        return self.seq(x)

class UpBlock(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(UpBlock, self).__init__()
        self.seq = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_dim,out_channels=out_dim,kernel_size=(1,2),stride=(1,2)),
            nn.Conv2d(in_channels=out_dim,out_channels=out_dim,kernel_size=3,stride=(2,2),padding=(1,1)),
            nn.BatchNorm2d(out_dim),
            nn.Conv2d(in_channels=out_dim,out_channels=out_dim,kernel_size=3,stride=(1,1),padding=(1,1)),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.seq(x)

class ConvGRUCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias, dtype):
        """
        Initialize the ConvLSTM cell
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int
            Number of channels of input tensor.
        :param hidden_dim: int
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param bias: bool
            Whether or not to add the bias.
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        """
        super(ConvGRUCell, self).__init__()
        self.height, self.width = input_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.dtype = dtype

        self.conv_gates = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                    out_channels=2*self.hidden_dim,  # for update_gate,reset_gate respectively
                                    kernel_size=kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

        self.conv_can = nn.Conv2d(in_channels=input_dim+hidden_dim,
                              out_channels=self.hidden_dim, # for candidate neural memory
                              kernel_size=kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).type(self.dtype))

    def forward(self, input_tensor, h_cur):
        """
        :param self:
        :param input_tensor: (b, c, h, w)
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w)
            current hidden and cell states respectively
        :return: h_next,
            next hidden state
        """
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([input_tensor, reset_gate*h_cur], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        return h_next

class ConvGRU(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 dtype, batch_first=False, bias=True, return_all_layers=False):
        """
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int e.g. 256
            Number of channels of input tensor.
        :param hidden_dim: int e.g. 1024
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param num_layers: int
            Number of ConvLSTM layers
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        :param alexnet_path: str
            pretrained alexnet parameters
        :param batch_first: bool
            if the first position of array is batch or not
        :param bias: bool
            Whether or not to add the bias.
        :param return_all_layers: bool
            if return hidden and cell states for all layers
        """
        super(ConvGRU, self).__init__()

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.dtype = dtype
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim[i - 1]
            cell_list.append(ConvGRUCell(input_size=(self.height, self.width),
                                         input_dim=cur_input_dim,
                                         hidden_dim=self.hidden_dim[i],
                                         kernel_size=self.kernel_size[i],
                                         bias=self.bias,
                                         dtype=self.dtype))

        # convert python list to pytorch module
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        :param input_tensor: (b, t, c, h, w) or (t,b,c,h,w) depends on if batch first or not
            extracted features from alexnet
        :param hidden_state:
        :return: layer_output_list, last_state_list
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list   = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                # input current hidden and cell state then compute the next hidden and cell state through ConvLSTMCell forward function
                h = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], # (b,t,c,h,w)
                                              h_cur=h)
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list   = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
