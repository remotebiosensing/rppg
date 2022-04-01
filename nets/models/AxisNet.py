import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange
from self_attention_cv.common import expand_to_batch
class AxisNet(nn.Module):
    def __init__(self,img_dim =(25,256)):
        super(AxisNet, self).__init__()
        #,img_dim,blocks,patch_dim,in_channels,out_channels
        self.vit_att_1 = BvpAttBlock(img_dim=(np.int(np.ceil(img_dim[0]/1)),np.int(np.ceil(img_dim[1]/1))),
                                     blocks=1,
                                     patch_dim=(np.int(np.ceil(img_dim[0]/1)),1),
                                     in_channels=3, out_channels=64)
        self.vit_att_2 = BvpAttBlock(img_dim=(np.int(np.ceil(img_dim[0] / 2)), np.int(np.ceil(img_dim[1] / 1))),
                                     blocks=1,
                                     patch_dim=(np.int(np.ceil(img_dim[0] / 2)), 1),
                                     in_channels=64, out_channels=128)
        self.vit_att_3 = BvpAttBlock(img_dim=(np.int(np.ceil(img_dim[0] / 4)), np.int(np.ceil(img_dim[1] / 1))),
                                     blocks=1,
                                     patch_dim=(np.int(np.ceil(img_dim[0] / 4)), 1),
                                     in_channels=128, out_channels=256)
        self.vit_att_4 = BvpAttBlock(img_dim=(np.int(np.ceil(img_dim[0] / 8)), np.int(np.ceil(img_dim[1] / 1))),
                                     blocks=1,
                                     patch_dim=(np.int(np.ceil(img_dim[0] / 8)), 1),
                                     in_channels=256, out_channels=512)

        self.up_scale_block = UpScaleBlock()

    def forward(self,x):
        bvp_in = x[0]
        y = self.vit_att_1(x[0])
        y = self.vit_att_2(y)
        y = self.vit_att_3(y)
        y = self.vit_att_4(y)
        y = self.up_scale_block(y)
        return y

class PTTBlock(nn.Module):
    def __init__(self):
        super(PTTBlock,self).__init__()
        self.seq = nn.Sequential(
            ConvBlock(3,64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
            UpBlock(512, 256),
            UpBlock(256, 128),
            UpBlock(128, 64),
            UpBlock(64, 1),
        )

    def forward(self,x):
        img_patches_ptt = rearrange(x, 'b c (patch_l l) h (patch_w w) -> b c h (l w) (patch_l patch_w)', patch_l = 2, patch_w = 128)
        img_patches_ptt = torch.mean(img_patches_ptt,dim=-1)
        max_ptt = img_patches_ptt.max(dim=-1,keepdim=True)[0]
        min_ptt = img_patches_ptt.min(dim=-1,keepdim=True)[0]
        img_patches_ptt = torch.mul(torch.sub(img_patches_ptt,min_ptt),1/(max_ptt-min_ptt))
        # [ batch , channel, h, lxw]
        # return self.seq(img_patches_ptt)
        return img_patches_ptt
class BVPBlock(nn.Module):
    def __init__(self):
        super(BVPBlock,self).__init__()
        self.vit_0 = ViT(img_dim=(25, 256), in_channels=3, blocks=1, patch_dim=(25, 1), classification=False)
        self.vit_1 = ViT(img_dim=(13, 128), in_channels=64, blocks=1, patch_dim=(13, 1), classification=False)
        self.vit_2 = ViT(img_dim=(7, 64), in_channels=128, blocks=1, patch_dim=(7, 1), classification=False)
        self.vit_3 = ViT(img_dim=(4, 32), in_channels=256, blocks=1, patch_dim=(4, 1), classification=False)


        self.d1 = ConvBlock(3, 64)
        self.d2 = ConvBlock(64, 128)
        self.d3 = ConvBlock(128, 256)
        self.d4 = ConvBlock(256, 512)
    def forward(self,x):
        vit = self.vit_0(x)
        vit = rearrange(vit,'b (x y) (patch_x patch_y c) -> b c (patch_x x) (patch_y y)',patch_x = 25,patch_y = 1,c = 3,x=1)
        y = self.d1(vit)
        vit = self.vit_1(y)
        vit = rearrange(vit,'b (x y) (patch_x patch_y c) -> b c (patch_x x) (patch_y y)',patch_x = 13,patch_y = 1,c = 64,x=1)
        y = self.d2(vit)
        vit = self.vit_2(y)
        vit = rearrange(vit,'b (x y) (patch_x patch_y c) -> b c (patch_x x) (patch_y y)',patch_x = 7,patch_y = 1,c = 128,x=1)
        y = self.d3(vit)
        vit = self.vit_3(y)
        vit = rearrange(vit, 'b (x y) (patch_x patch_y c) -> b c (patch_x x) (patch_y y)', patch_x=4, patch_y=1, c=256,
                        x=1)
        y = self.d4(vit)


        return self.up(y)
        # return self.seq(img_patches_bvp)
class BvpAttBlock(nn.Module):
    def __init__(self,img_dim,blocks,patch_dim,in_channels,out_channels):
        super(BvpAttBlock, self).__init__()
        self.bvpattblock= nn.Sequential(
            ViT(img_dim=img_dim, in_channels=in_channels, blocks=blocks, patch_dim=patch_dim),
            ConvBlock(in_dim=in_channels, out_dim=out_channels)
        )
    def forward(self,x):
        return self.bvpattblock(x)


class UpScaleBlock(nn.Module):
    def __init__(self):
        super(UpScaleBlock, self).__init__()
        self.up = nn.Sequential(
            UpBlock(512, 256),
            UpBlock(256, 128),
            UpBlock(128, 64),
            UpBlock(64, 1),
        )
    def forward(self,x):
        return self.up(x)

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
            nn.Conv2d(in_channels=out_dim,out_channels=out_dim,kernel_size=3,stride=(2,1),padding=(1,1)),
            nn.BatchNorm2d(out_dim),
            nn.Conv2d(in_channels=out_dim,out_channels=out_dim,kernel_size=3,stride=1,padding=(1,1)),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.seq(x)

class ConvBlock_main(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ConvBlock_main, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=(4, 8), stride=(4, 8),padding=0),
            nn.BatchNorm2d(out_dim),
            nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=(4, 6), stride=(4, 6),padding=0),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
        # (25 - 3)/2 +1

    def forward(self, x):
        return self.seq(x)

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
                 dim=512,
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
        self.dim = self.token_dim
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head

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
