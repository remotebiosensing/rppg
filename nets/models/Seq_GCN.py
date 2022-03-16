import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from self_attention_cv import AxialAttentionBlock,MultiHeadSelfAttention,ViT,TransformerEncoder
from self_attention_cv.common import expand_to_batch
from self_attention_cv.pos_embeddings import AbsPosEmb1D, RelPosEmb2D
from nets.models.gcn_utils import KNN_dist, View_selector, LocalGCN, NonLocalMP
from nets.models import PhysNet
from einops import rearrange
import math
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=32):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class channel_Encoding_Block(nn.Module):
    def __init__(self,token=32,dim_head=32):
        super(channel_Encoding_Block, self).__init__()
        #x-axis
        self.emb_x_r = AbsPosEmb1D(tokens=token,dim_head=32)
        self.muulti_x_r = MultiHeadSelfAttention(dim=32)

        self.emb_x_g = AbsPosEmb1D(tokens=token,dim_head=32)
        self.muulti_x_g = MultiHeadSelfAttention(dim=32)

        self.emb_x_b = AbsPosEmb1D(tokens=token,dim_head=32)
        self.muulti_x_b = MultiHeadSelfAttention(dim=32)

        self.norm1_x = nn.LayerNorm(32)
        #y-axis
        self.emb_y_r = AbsPosEmb1D(tokens=token,dim_head=32)
        self.muulti_y_r = MultiHeadSelfAttention(dim=32)

        self.emb_y_g = AbsPosEmb1D(tokens=token,dim_head=32)
        self.muulti_y_g = MultiHeadSelfAttention(dim=32)

        self.emb_y_b = AbsPosEmb1D(tokens=token,dim_head=32)
        self.muulti_y_b = MultiHeadSelfAttention(dim=32)

        self.norm1_y = nn.LayerNorm(32)
        #FNN
        self.norm1 = nn.LayerNorm(32)
        self.norm2 = nn.LayerNorm(32)
        self.norm3 = nn.LayerNorm(32)

        self.drop_out = nn.Dropout2d(0.5)

        self.linear_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*6, 2048),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(2048,1024),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 32)
        )
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        [batch, channel, length, width, height] = x.shape
        ori = x
        tst = torch.permute(x, [0, 1, 4, 2, 3])  # tst1
        x_tmp = []
        for i in range(width):
            inner_tmp = []

            for j in range(3):
                in_ = torch.unsqueeze(tst[:, j, i, :, :], dim=1)
                if j == 0:
                    emb_y = self.emb_y_r(in_)
                    emb_y = torch.squeeze(emb_y)
                    tmp = self.muulti_y_r(emb_y)
                elif j == 1:
                    emb_y = self.emb_y_g(in_)
                    emb_y = torch.squeeze(emb_y)
                    tmp = self.muulti_y_g(emb_y)
                else:
                    emb_y = self.emb_y_b(in_)
                    emb_y = torch.squeeze(emb_y)
                    tmp = self.muulti_y_b(emb_y)


                tmp = emb_y + self.drop_out(tmp)
                tmp = self.norm1_y(tmp)

                tmp = torch.unsqueeze(tmp, dim=1)
                # tmp = self.conv1x1_2(tmp)
                if j == 0:
                    inner_tmp = tmp
                else:
                    inner_tmp = torch.cat([inner_tmp, tmp], dim=1)
            if i == 0:
                x_tmp = torch.unsqueeze(inner_tmp, dim=2)
            else:
                x_tmp = torch.cat([x_tmp, torch.unsqueeze(inner_tmp, dim=2)], dim=2)

        # x_tmp = x_tmp + self.norm1_y(tst)
        # [0, 1, 4, 2, 3]
        x_tmp = torch.permute(x_tmp, [0, 1, 3, 4, 2])  # tst2
        # x_tmp = torch.permute(x_tmp, [0, 1, 3, 2, 4])

        x = torch.sigmoid(x_tmp)*x

        x = torch.permute(x, [0, 1, 3, 4, 2])  # x1
        y = []
        for i in range(width):
            inner_tmp = []

            for j in range(3):
                in_ = torch.unsqueeze(x[:, j, i, :, :], dim=1)

                if j == 0:
                    emb_x = self.emb_x_r(in_)
                    emb_x = torch.squeeze(emb_y)
                    tmp = self.muulti_x_r(emb_y)
                elif j == 1:
                    emb_x = self.emb_x_g(in_)
                    emb_x = torch.squeeze(emb_y)
                    tmp = self.muulti_x_g(emb_y)
                else:
                    emb_x = self.emb_x_b(in_)
                    emb_x = torch.squeeze(emb_y)
                    tmp = self.muulti_x_b(emb_y)

                tmp = emb_x + self.drop_out(tmp)
                tmp = self.norm1_x(tmp)

                tmp = torch.unsqueeze(tmp, dim=1)
                # tmp = self.conv1x1_2(tmp)
                if j == 0:
                    inner_tmp = tmp
                else:
                    inner_tmp = torch.cat([inner_tmp, tmp], dim=1)
            if i == 0:
                y = torch.unsqueeze(inner_tmp, dim=2)
            else:
                y = torch.cat([y, torch.unsqueeze(inner_tmp, dim=2)], dim=2)

        # y = y + self.norm1(x)
        x = torch.permute(x,[0,1,4,2,3])
        y = torch.permute(y, [0, 1, 4, 2, 3])  # x2
        # x_tmp = torch.permute(x_tmp,[0,1,3,2,4])
        # y = torch.relu(y)
        # x_tmp = torch.relu(x_tmp)
        y = torch.sigmoid(y) * x
        # y = ori*torch.relu(x_tmp+y)

        # # y = self.linear_net(y)
        # y = y + self.drop_out(y)
        # att = self.norm2(y)
        # y = self.linear_net(att)
        # y = att + self.drop_out(y)
        # y = self.norm3(y)

        return y

class Seq_GCN(nn.Module):
    def __init__(self, roi = 32, divide = 4, length = 32):
        super().__init__()
        self.Encoding_block = channel_Encoding_Block()



        # self.adaptivepool = nn.AdaptiveMaxPool3d([32,1,1])
        # self.adaptivepool_mean = nn.AdaptiveAvgPool2d(1)

        self.adaptivepool = nn.AdaptiveMaxPool3d([32, 1, 1])
        self.adaptivepool_mean = nn.AdaptiveAvgPool2d(1)

        self.linear_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(24*24*32 , 1024),
            nn.Dropout(0.5),
            nn.ELU(inplace=True),
            # nn.Linear(2048, 1024),
            # nn.Dropout(0.5),
            # nn.ELU(inplace=True),
            nn.Linear(1024, 32)
        )

        self.bpm_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(24*24*32, 512),
            nn.BatchNorm1d(512),
            # nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(512,128),
            nn.BatchNorm1d(128),
            # nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),

        )
        self.sigmoid = nn.Sigmoid()
        self.drop_out = nn.Dropout(0.5)
        self.norm = nn.LayerNorm(32)

        #VIT test
        self.token_dim_w = 3 * (32 **2)
        self.dim_w = 3
        self.project_patches_w = nn.Linear(self.token_dim_w,self.dim_w)

        self.transformer_w = TransformerEncoder(3, blocks=6, heads=4,
                                              dim_head=4,
                                              dim_linear_block=3,
                                              dropout=0.2)

        self.cls_token_w = nn.Parameter(torch.randn(1, 1, 3))
        self.pos_emb1D_w = nn.Parameter(torch.randn(32 + 1, 3))
        self.emb_dropout_w = nn.Dropout(0.2)

        #VIT test
        self.token_dim_h = 3 * (32 **2)
        self.dim_h = 3
        self.project_patches_h = nn.Linear(self.token_dim_h,self.dim_h)

        self.transformer_h = TransformerEncoder(3, blocks=6, heads=4,
                                              dim_head=4,
                                              dim_linear_block=3,
                                              dropout=0.2)

        self.cls_token_h = nn.Parameter(torch.randn(1, 1, 3))
        self.pos_emb1D_h = nn.Parameter(torch.randn(32 + 1, 3))
        self.emb_dropout_h = nn.Dropout(0.2)

        #VIT test
        self.token_dim = 3 * (32 **2)
        self.dim = 3
        self.project_patches = nn.Linear(self.token_dim,self.dim)

        self.transformer = TransformerEncoder(3, blocks=6, heads=4,
                                              dim_head=4,
                                              dim_linear_block=3,
                                              dropout=0.2)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 3))
        self.pos_emb1D = nn.Parameter(torch.randn(32 + 1, 3))
        self.emb_dropout = nn.Dropout(0.2)

        self.dropout = nn.Dropout2d(0.5)

        # self.sequential = nn.Sequential(
        #     nn.Linear(32*256,1024),
        #     nn.Linear(1024,256),
        #     nn.Linear
        # )
        self.normlayer = nn.LayerNorm(32)
        self.conv3d = nn.Conv3d(in_channels=3,out_channels=1,kernel_size=(1,5,5))
        self.conv2d = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5)
        self.gelu = nn.GELU()


    def forward(self,x):
        # x = F.normalize(x,dim=2)
        img_patches_w = rearrange(x,'b c length w h -> b w (length h c)')
        batch_size, tokens, _ = img_patches_w.shape
        img_patches_w = self.project_patches_w(img_patches_w)
        img_patches_w = torch.cat((expand_to_batch(self.cls_token_w, desired_size=batch_size), img_patches_w), dim=1)

        # add pos. embeddings. + dropout
        # indexing with the current batch's token length to support variable sequences
        img_patches_w = img_patches_w + self.pos_emb1D_w[:tokens + 1, :]
        patch_embeddings_w = self.emb_dropout_w(img_patches_w)
        w = self.transformer_w(patch_embeddings_w)
        w = w[:, 1:, :]
        w = rearrange(w,'b w (length h c) -> b (c length w h)', w=32, length = 1,c = 3)

        w = self.w_sequential(w)
        w = rearrange(w,'b (c length w h) -> b c length w h', w=32, length = 1,c = 3)

        img_patches_h = rearrange(x,'b c length w h -> b h (length w c)')
        batch_size, tokens, _ = img_patches_h.shape
        img_patches_h = self.project_patches_h(img_patches_h)
        img_patches_h = torch.cat((expand_to_batch(self.cls_token_h, desired_size=batch_size), img_patches_h), dim=1)

        # add pos. embeddings. + dropout
        # indexing with the current batch's token length to support variable sequences
        img_patches_h = img_patches_h + self.pos_emb1D_h[:tokens + 1, :]
        patch_embeddings_h = self.emb_dropout_h(img_patches_h)
        h = self.transformer_h(patch_embeddings_h)
        h = h[:, 1:, :]
        h = rearrange(h, 'b h (length w c) -> b (c length w h)', h=32, length=1, c=3)

        h = self.h_sequential(h)
        h = rearrange(h,'b (c length w h) -> b c length w h', h=32, length = 1,c = 3)


        att = x*self.softmax(w*h)


        # img_patches = rearrange(x,'b c length w h -> b length (h w c)')
        # batch_size, tokens, _ = img_patches.shape
        # img_patches = self.project_patches(img_patches)
        # img_patches = torch.cat((expand_to_batch(self.cls_token, desired_size=batch_size), img_patches), dim=1)
        #
        # # add pos. embeddings. + dropout
        # # indexing with the current batch's token length to support variable sequences
        # img_patches = img_patches + self.pos_emb1D[:tokens + 1, :]
        # patch_embeddings = self.emb_dropout_h(img_patches)
        # i = self.transformer(patch_embeddings)
        # i = i[:, 1:, :]
        # i = rearrange(i, 'b length (h w c) -> b (c length w h)', h=1, length=32, c=3)
        #
        # i = self.l_sequential(i)
        # i = rearrange(i,'b (c length w h) -> b c length w h', h=1, length = 32,c = 3)

        # att = torch.matmul(w,h)
        # att = torch.squeeze(att)
        # att = self.softmax2d(att)
        # att = torch.unsqueeze(att,dim=2)
        # att = i * att
        # att = self.normlayer(x * att)




        x = self.conv3d(att)
        x = torch.squeeze(x)
        x = self.conv2d(x)
        x = self.gelu(x)


        # y = rearrange(y,'b w (length h) -> b (length w h)',w=32,h=32)
        # att = self.Encoding_block(x)

        # max_y = self.adaptivepool(att)
        # mean_y = self.adaptivepool_mean(att)
        #
        #
        # max_y = torch.squeeze(max_y)
        # mean_y = torch.squeeze(mean_y)
        # y = torch.cat([mean_y, max_y], axis=1)

        bpm = self.bpm_net(x)
        bpm = 200 * self.sigmoid(bpm)

        # y = torch.cat([mean_y,max_y],axis=1)1
        y = self.linear_net(x)
        # y = y + self.drop_out(y)
        # y = self.norm(y)
        # bpm = []
        return y,bpm,att#p.view(-1,self.length-2),p.view(-1,self.length-2)#y.view(-1,self.length-6) # 32 length




# class Seq_GCN(nn.Module):
#     def __init__(self):
#         super(Seq_GCN, self).__init__()
#     def forward(self,x):
#         x = rearrange(x,'b c length w h => b length (w h c)')
#         print("A")
#         y = []
#         bpm = []
#         att = []
#         return y,bpm,att

#temporal