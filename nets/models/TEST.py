import torch
import torch.nn as nn
from nets.modules.vit_pytorch.max_vit import MaxViT, MaxVit_GAN_layer, MaxViT_layer, CrissCrossAttention, FeedForward, \
    MBConv
from einops import rearrange
from einops.layers.torch import Rearrange
import math
import torch.nn.functional as F
from julius.lowpass import LowPassFilter
import numpy as np
from torch import Tensor
from params import params
from torchvision import transforms
import timm

class APNET(nn.Module):
    def __init__(self,  num_filters=64, kernel_size=3):
        super(APNET, self).__init__()

        self.f_m = APNET_Backbone()
        self.l_m = APNET_Backbone()
        self.r_m = APNET_Backbone()
        # Feature fusion 레이어
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=16, kernel_size=1),
            nn.ReLU()
        )

        # Attention 메커니즘을 위한 추가 레이어
        self.attention = nn.Sequential(
            nn.Linear(3, 3),
            nn.Softmax(dim=1)
        )

        self.sub = sub()

    def forward(self, x):
        f_feature_map = self.f_m.main_seq_stem(x[0])
        l_feature_map = self.l_m.main_seq_stem(x[1])
        r_feature_map = self.r_m.main_seq_stem(x[2])

        f_s = self.sub(torch.mean(x[0], dim=(3,4)))
        l_s = self.sub(torch.mean(x[1], dim=(3,4)))
        r_s = self.sub(torch.mean(x[2], dim=(3,4)))

        f_feature_map = (1+f_s)*f_feature_map
        l_feature_map = (1+l_s)*l_feature_map
        r_feature_map = (1+r_s)*r_feature_map

        # (Batch, Channel, Time_Length, height*width) 형태로 변경
        combined_feature_maps = torch.cat([f_feature_map, l_feature_map, r_feature_map], dim=1)

        fused_feature_maps = self.feature_fusion(combined_feature_maps)

        # 각 Backbone에 공유된 feature map을 전달
        f_out = self.f_m.forward_after_stem(fused_feature_maps)
        l_out = self.l_m.forward_after_stem(fused_feature_maps)
        r_out = self.r_m.forward_after_stem(fused_feature_maps)
        # (Batch, Time_Length, 3) 형태로 변경
        combined = torch.stack([f_out, l_out, r_out], dim=2)

        # Attention 메커니즘 적용
        attention_weights = self.attention(torch.mean(combined, dim=1))
        attention_weights = attention_weights.unsqueeze(1)

        # 각 출력에 가중치를 곱한 후 합산
        out = torch.sum(attention_weights * combined, dim=2)
        # out = (out - torch.mean(out)) / torch.std(out)
        return out


class sub(nn.Module):
    def __init__(self, input_channels=3):
        super(sub, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        x = self.conv1(x)
        x = self.sigmoid(x)

        return x.unsqueeze(2).expand(-1, -1, params.time_length, -1)

class APNET_Backbone(nn.Module):
    def __init__(self):
        super(APNET_Backbone, self).__init__()

        self.dog_transform = transforms.Compose([
            transforms.Lambda(lambda x: x.float() / 255.0),
            transforms.Lambda(lambda x: x[1] - x[0]),  # Gaussian 차이
            transforms.Lambda(lambda x: torch.abs(x)),  # 절대값 취하기
            transforms.Normalize([0.5], [0.5]),  # -1 ~ 1 사이의 값으로 정규화
        ])

        self.main_seq_stem = nn.Sequential(
            Rearrange('b l c h w -> (b l) c h w'),
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.25),
            Rearrange('(b l) c h w -> b c l (h w)', l=params.time_length)
        )
        # padding =  kernel /2
        # layerdim = dim * expand 대신 ㅅㅏ용
        self.main_seq_max_1 = nn.Sequential(
            MaxViT_layer(layer_depth=1, layer_dim_in=16, layer_dim=32,
                         kernel=(2, 16), dilation=(1, 8), padding=(1,8),
                         mbconv_expansion_rate=4, mbconv_shrinkage_rate=0.5, w=4, dim_head=8, dropout=0.1, flag=False)
        )
        self.sa_main = SpatialAttention(in_channels=32)
        self.rnn = nn.LSTM(input_size=64*32, hidden_size=32, num_layers=1, batch_first=True, bidirectional=True)
        self.adaptive = nn.AdaptiveAvgPool2d((32, 16))
        self.max_vit = MaxViT_layer(layer_depth=2, layer_dim_in=1, layer_dim=32,
                                    kernel=3, dilation=1, padding=1,
                                    mbconv_expansion_rate=4, mbconv_shrinkage_rate=0.25, w=4, dim_head=32, dropout=0.1,
                                    flag=False)
        self.be_conv1d = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding="same")
        self.out_conv1d = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

        # self.init_weights()

    def forward(self,x):
        diff_x = []
        for i in range(x.shape[1]):
            frame_diff = self.dog_transform([x[:, i, :, :, :], x[:, i + 1, :, :, :]])
            diff_x.append(frame_diff)

        diff_x = torch.stack(diff_x, dim=1)  # (batch_size, num_frames-1, channel, height, width)
        diff_x = diff_x.permute(0, 1, 3, 4, 2)  # (batch_size, num_frames-1, height, width, channel)
        diff_x = diff_x.reshape(-1, self.num_frames - 1, diff_x.shape[2], diff_x.shape[3], diff_x.shape[4])

        feature_map = self.main_seq_stem(x)
        # 차이 계산된 이미지들을 합치기
        x = torch.cat(diff_x, dim=1)  # (batch_size, num_frames-1, channel,

        out = self.forward_after_stem(feature_map)
        return out

    def forward_after_stem(self, feature_map):
        main_2 = self.main_seq_max_1(feature_map)
        main_3 = self.sa_main(main_2)

        # Rearrange to feed into LSTM
        main_4 = rearrange(main_3, 'b c l hw -> b l (c hw)')
        rnn_out, _ = self.rnn(main_4)

        # Rearrange back to original shape

        main_5 = rearrange(rnn_out, 'b l (c h w) -> b c l (h w)', c=1, h=8, w=8)
        out_1 = self.max_vit(main_5)
        out_2 = torch.squeeze(out_1)
        out_3 = torch.mean(out_2, dim=-1)

        out_att = self.be_conv1d(out_3)
        out_4 = (1 + self.sigmoid(out_att)) * out_3
        out_5 = self.out_conv1d(out_4)
        out = torch.squeeze(out_5)
        out = (out - torch.mean(out)) / torch.std(out)
        return out



# TMP
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel=3):
#         super(SpatialAttention, self).__init__()
#
#         self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel, padding=kernel // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#
#         return self.sigmoid(x)
class SpatialAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(SpatialAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // reduction_ratio, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels // reduction_ratio, out_channels=in_channels, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.conv2(self.conv1(self.avg_pool(x)))
        max_out = self.conv2(self.conv1(self.max_pool(x)))

        out = self.sigmoid(avg_out + max_out)
        return x * out


class VideoTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, output_dim):
        super(VideoTransformer, self).__init__()

        self.hidden_dim = hidden_dim

        # Input embedding layer
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
            for _ in range(num_layers)
        ])

        # Output layers
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # [B, 32, L, W, H]
        b, c, l, _, _ = x.shape
        x = torch.reshape(x, (b, l, -1))
        # x has shape (batch_size, num_frames, input_dim)
        # Embed the input
        x = self.embedding(x)

        # Transpose x to shape (num_frames, batch_size, hidden_dim)
        x = x.transpose(0, 1)

        # Apply the Transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        # Average pool over the frames
        x = x.transpose(0, 1)  # Transpose back to shape (batch_size, num_frames, hidden_dim)
        x = self.avg_pool(x.transpose(1, 2)).squeeze(-1)  # Shape: (batch_size, hidden_dim)

        # Map to the output space
        x = self.fc(x)  # Shape: (batch_size, output_dim)

        return x
class VideoTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, output_dim):
        super(VideoTransformer, self).__init__()

        self.hidden_dim = hidden_dim

        # Input embedding layer
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim)
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
            for _ in range(num_layers)
        ])

        # Output layers
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x has shape (batch_size, num_frames, input_dim)
        b, _, l, _, _ = x.shape
        x = torch.permute(x, (0, 2, 1, 3, 4))
        x = torch.reshape(x, (b, l, -1))
        # Embed the input
        x = self.embedding(x)
        x = self.pos_enc(x)

        # Transpose x to shape (num_frames, batch_size, hidden_dim)
        x = x.transpose(0, 1)

        # Apply the Transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        # Average pool over the frames
        x = x.transpose(0, 1)  # Transpose back to shape (batch_size, num_frames, hidden_dim)
        x = self.avg_pool(x.transpose(1, 2)).squeeze(-1)  # Shape: (batch_size, hidden_dim)

        # Map to the output space
        x = self.fc(x)  # Shape: (batch_size, output_dim)

        return x
class VideoRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(VideoRNN, self).__init__()

        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x has shape (batch_size, num_frames, input_dim)
        b, feature = x.shape
        x = torch.reshape(x, (b, -1, feature))
        # Apply the RNN
        _, (h_n, _) = self.rnn(x)

        # Use the last hidden state as input to the final FC layer
        x = h_n[-1]
        x = self.fc(x)

        return x
class VideoRNNAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.0):
        super(VideoRNNAttention, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        self.attention = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        b, feature = x.shape
        x = torch.reshape(x, (b, -1, feature))

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, (hn, cn) = self.rnn(x, (h0, c0))

        # Compute attention scores
        attn_scores = self.attention(out)

        # Apply softmax to get attention weights
        attn_weights = torch.softmax(attn_scores, dim=1)

        # Apply attention weights to hidden states
        attn_out = torch.sum(attn_weights * out, dim=1)

        # Pass attention output through linear layer to get final prediction
        out = self.out(attn_out)

        return out
class VideoTransformer2(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads):
        super(VideoTransformer2, self).__init__()

        # Embedding layer
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.pos_enc = PositionalEncoding(hidden_dim)

        # Transformer layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(TransformerLayer(hidden_dim, num_heads))

        # Output layer
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x has shape (batch, length, input_dim)
        x = self.embedding(x)
        x = self.pos_enc(x)

        for layer in self.layers:
            x = layer(x)

        # x = x.mean(dim=1)  # Global average pooling

        x = self.output(x)
        return x
class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_length=1000):
        super(PositionalEncoding, self).__init__()
        self.hidden_dim = hidden_dim

        # Compute positional encodings for the max_length
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))
        pe = torch.zeros(max_length, hidden_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # Register the positional encodings as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add the positional encodings to the input
        x = x + self.pe[:, :x.size(1)]
        return x
class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(TransformerLayer, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = self.norm1(x)

        # Feedforward
        ff_output = self.feedforward(x)
        x = x + ff_output
        x = self.norm2(x)

        return x
class MHA(nn.Module):
    def __init__(self, input_size, num_heads, hidden_size):
        super(MHA, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.linear_in = nn.Linear(input_size, hidden_size)
        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads)
        self.linear_out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x is a tensor of shape (batch_size, seq_len, input_size)
        # x is a tensor of shape (batch_size, input_size, seq_len)
        x = x.permute(2, 0, 1)  # shape (seq_len, batch_size, input_size)
        x = self.linear_in(x)  # shape (seq_len, batch_size, hidden_size)
        x, _ = self.multihead_attn(x, x, x)  # shape (seq_len, batch_size, hidden_size)
        x = self.linear_out(x)  # shape (seq_len, batch_size, 1)
        x = x.squeeze(-1)  # shape (seq_len, batch_size)
        return x
class MultiHeadAttention1d(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention1d, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        # Q, K, V linear projections for each head
        self.w_qs = nn.Linear(d_model, d_model * num_heads, bias=False)
        self.w_ks = nn.Linear(d_model, d_model * num_heads, bias=False)
        self.w_vs = nn.Linear(d_model, d_model * num_heads, bias=False)

        # Final linear projection for each head
        self.w_out = nn.Linear(d_model * num_heads, d_model, bias=False)

    def forward(self, q,k,v):
        # Input shape: (batch_size, seq_len, d_model)
        b,s = q.shape
        q = torch.reshape(q,(b,s,1))
        k = torch.reshape(k, (b, s, 1))
        v = torch.reshape(v, (b, s, 1))


        batch_size, seq_len, d_model = q.size()

        # Linear projections for each head
        q = self.w_qs(q).view(batch_size, seq_len, self.num_heads, self.d_model).transpose(1,
                                                                                           2)  # (batch_size, num_heads, seq_len, d_model)
        k = self.w_ks(k).view(batch_size, seq_len, self.num_heads, self.d_model).transpose(1,
                                                                                           2)  # (batch_size, num_heads, seq_len, d_model)
        v = self.w_vs(v).view(batch_size, seq_len, self.num_heads, self.d_model).transpose(1,
                                                                                           2)  # (batch_size, num_heads, seq_len, d_model)

        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (
                    self.d_model ** 0.5)  # (batch_size, num_heads, seq_len, seq_len)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)  # (batch_size, num_heads, seq_len, d_model)

        # Concatenate attention heads and apply final linear projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len,
                                                            -1)  # (batch_size, seq_len, d_model * num_heads)
        output = self.w_out(context)  # (batch_size, seq_len, d_model)

        return output
class TEST2(nn.Module):
    def __init__(self, ver):
        super(TEST2, self).__init__()

        length = 32
        height, width = (128, 128)
        self.ver = ver

        self.main_conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1)
        self.main_conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1)
        self.main_seqmax = MaxViT_layer(layer_depth=2, layer_dim_in=3, layer_dim=32,
                                        kernel=(1, 32), dilation=(1, 32), padding=0,
                                        mbconv_expansion_rate=4, mbconv_shrinkage_rate=0.25, w=4, dim_head=32,
                                        dropout=0.1, flag=False)

        self.ptt_conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1)
        self.ptt_conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1)
        self.ptt_seqmax = MaxViT_layer(layer_depth=2, layer_dim_in=3, layer_dim=32,
                                       kernel=(1, 8), dilation=(1, 32), padding=0,
                                       mbconv_expansion_rate=4, mbconv_shrinkage_rate=0.25, w=4, dim_head=32,
                                       dropout=0.1, flag=True)

        self.bvp_conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1)
        self.bvp_conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1)
        self.bvp_seqmax = MaxViT_layer(layer_depth=2, layer_dim_in=3, layer_dim=32,
                                       kernel=(1, 8), dilation=(1, 32), padding=0,
                                       mbconv_expansion_rate=4, mbconv_shrinkage_rate=0.25, w=4, dim_head=32,
                                       dropout=0.1, flag=True)

        self.sa_main = SpatialAttention()
        self.sa_bvp = SpatialAttention()
        self.sa_ptt = SpatialAttention()

        self.adaptive = nn.AdaptiveAvgPool2d((32, 16))

        self.max_vit = MaxViT_layer(layer_depth=2, layer_dim_in=1, layer_dim=32,
                                    kernel=3, dilation=1, padding=1,
                                    mbconv_expansion_rate=4, mbconv_shrinkage_rate=0.25, w=4, dim_head=32, dropout=0.1,
                                    flag=False)

        self.be_conv1d = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding="same")
        self.out_conv1d = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

        self.conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1)

    def forward(self, x):
        length = 32

        m0 = torch.permute(x, (0, 2, 1, 3, 4))
        m1 = rearrange(m0, 'b l c h w -> (b l) c h w')  # 1
        m2 = self.main_conv1(m1)  # 2
        m3 = self.main_conv2(m2)  # 3
        m4 = rearrange(m3, '(b l) c h w -> b l c h w', l=length)  # 4
        m5 = torch.permute(m4, (0, 2, 1, 3, 4))  # 5
        m6 = rearrange(m5, 'b c l h w-> b c l (h w)')  # 6
        m7 = self.main_seqmax(m6)  # 7
        m8 = self.sa_main(m7)  # 8
        m9 = self.adaptive(m8)  # 9
        m10 = rearrange(m9, 'b c l (h w) -> b c l h w', h=4, w=4)  # 10
        # main_11 = torch.permute(main_10,(0, 2, 1, 3, 4))

        p0 = torch.permute(x, (0, 3, 1, 2, 4))  # 0
        p1 = rearrange(p0, 'b h c l w -> (b h) c l w')  # 1
        p2 = self.ptt_conv1(p1)  # 2
        p3 = self.ptt_conv2(p2)  # 3
        p4 = rearrange(p3, '(b h) c l w -> b h c l w', h=128)  # 4
        p5 = torch.permute(p4, (0, 2, 1, 3, 4))  # 5
        p6 = rearrange(p5, 'b c h l w -> b c h (l w)')  # 6
        p7 = self.ptt_seqmax(p6)  # 7
        p8 = self.sa_ptt(p7)  # 8
        p9 = rearrange(p8, 'b c h (l w) -> b c h l w', l=4, w=4)  # 9
        p10 = torch.permute(p9, (0, 1, 3, 2, 4))  # 10

        b0 = torch.permute(x, (0, 4, 1, 2, 3))
        b1 = rearrange(b0, 'b w c l h -> (b w) c l h')
        b2 = self.bvp_conv1(b1)
        b3 = self.bvp_conv2(b2)
        b4 = rearrange(b3, '(b w) c l h -> b w c l h', w=128)
        b5 = torch.permute(b4, (0, 2, 1, 3, 4))
        b6 = rearrange(b5, 'b c w l h -> b c w (l h)')
        b7 = self.bvp_seqmax(b6)
        b8 = self.sa_bvp(b7)
        b9 = rearrange(b8, 'b c w (l h) -> b c w l h', l=4, h=4)
        b10 = torch.permute(b9, (0, 1, 3, 4, 2))

        if self.ver == 0:  # M(W@H)+M
            att = b10 @ p10
            m11 = m10 * F.interpolate(att, scale_factor=(8, 1, 1)) + m10
        elif self.ver == 1:  # M(W+H)+M
            att1 = F.interpolate(b10, scale_factor=(1, 1, 1 / 16))  # w
            att2 = F.interpolate(p10, scale_factor=(1, 1 / 16, 1))  # H
            att = att1 + att2
            m11 = m10 * F.interpolate(att, scale_factor=(8, 1, 1)) + m10
        elif self.ver == 2:  # MW+M
            att = F.interpolate(b10, scale_factor=(1, 1, 1 / 16))  # w
            m11 = m10 * F.interpolate(att, scale_factor=(8, 1, 1)) + m10
        elif self.ver == 3:  # MH+M
            att2 = F.interpolate(p10, scale_factor=(1, 1 / 16, 1))  # H
            att = att2
            m11 = m10 * F.interpolate(att, scale_factor=(8, 1, 1)) + m10
        elif self.ver == 4:  # M
            m11 = m10
        elif self.ver == 5:  # WM + HM
            b11 = F.interpolate(b10, scale_factor=(8, 1, 1 / 16))  # W
            p11 = F.interpolate(p10, scale_factor=(8, 1 / 16, 1))  # H
            m11 = m10 * b11 + m10 * p11
        elif self.ver == 6:  # WM
            b11 = F.interpolate(b10, scale_factor=(8, 1, 1 / 16))  # W
            m11 = m10 * b11
        elif self.ver == 7:  # HM
            p11 = F.interpolate(p10, scale_factor=(8, 1 / 16, 1))  # H
            m11 = m10 * p11
        elif self.ver == 8:  # H
            p11 = F.interpolate(p10, scale_factor=(8, 1 / 16, 1))  # H
            m11 = p11
        elif self.ver == 9:  # W
            b11 = F.interpolate(b10, scale_factor=(8, 1, 1 / 16))  # W
            m11 = b11

        m13 = rearrange(m11, 'b c l w h -> b c l (w h)')  # 12

        o1 = self.conv(m13)
        # o1 = self.max_vit(m13)                                             #13
        o2 = torch.squeeze(o1)  # 14
        o3 = torch.mean(o2, dim=-1)  # 15
        out_att = self.be_conv1d(o3)
        o4 = (1 + self.sigmoid(out_att)) * o3  # 16
        o5 = self.out_conv1d(o4)  # 17
        out = torch.squeeze(o5)

        return out
class TEST(nn.Module):
    def __init__(self):
        super(TEST, self).__init__()

        length = 32
        height, width = (128, 128)

        self.main_seq_stem = nn.Sequential(
            Rearrange('b c l h w -> (b l) c h w'),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1),
            Rearrange('(b l) c h w -> b c l (h w)', l=length)
        )
        self.main_seq_max_1 = nn.Sequential(
            MaxViT_layer(layer_depth=2, layer_dim_in=3, layer_dim=32,
                         kernel=(1, 32), dilation=(1, 32), padding=0,
                         mbconv_expansion_rate=4, mbconv_shrinkage_rate=0.25, w=4, dim_head=32, dropout=0.1, flag=False)
        )

        self.ptt_seq_stem = nn.Sequential(
            Rearrange('b c l h w -> (b h) c l w'),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1),
            Rearrange('(b h) c l w -> b c h (l w)', h=height)
        )
        self.ptt_seq_max_1 = nn.Sequential(
            MaxViT_layer(layer_depth=2, layer_dim_in=3, layer_dim=32,
                         kernel=(1, 8), dilation=(1, 32), padding=0,
                         mbconv_expansion_rate=4, mbconv_shrinkage_rate=0.25, w=4, dim_head=32, dropout=0.1, flag=True))

        self.bvp_seq_stem = nn.Sequential(
            Rearrange('b c l h w -> (b w) c l h'),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1),
            Rearrange('(b w) c l h -> b c w (l h)', w=width)
        )
        self.bvp_seq_max_1 = nn.Sequential(
            MaxViT_layer(layer_depth=2, layer_dim_in=3, layer_dim=32,
                         kernel=(1, 8), dilation=(1, 32), padding=0,
                         mbconv_expansion_rate=4, mbconv_shrinkage_rate=0.25, w=4, dim_head=32, dropout=0.1, flag=True))
        self.max_vit = MaxViT_layer(layer_depth=2, layer_dim_in=1, layer_dim=32,
                                    kernel=3, dilation=1, padding=1,
                                    mbconv_expansion_rate=4, mbconv_shrinkage_rate=0.25, w=4, dim_head=32, dropout=0.1,
                                    flag=False)
        self.adaptation = nn.AdaptiveAvgPool2d((4, 2))

        self.sa_main = SpatialAttention()
        self.sa_bvp = SpatialAttention()
        self.sa_ptt = SpatialAttention()

        self.adaptive = nn.AdaptiveAvgPool2d((32, 16))
        self.be_conv1d = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding="same")
        self.out_conv1d = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        main_1 = self.main_seq_stem(x)
        main_2 = self.main_seq_max_1(main_1)
        # ver1
        main_3 = self.sa_main(main_2)
        # ver2
        # main_att = self.sa_main(main)
        # main = main_att*main + main
        main_4 = self.adaptive(main_3)
        main_5 = rearrange(main_4, 'b c l (w h) -> b c l w h', w=4, h=4)
        # main = self.main_seq_max_2(main)

        bvp_1 = self.bvp_seq_stem(x)
        bvp_2 = self.bvp_seq_max_1(bvp_1)
        # ver1
        bvp_3 = self.sa_bvp(bvp_2)
        # ver2
        # bvp_att = self.sa_bvp(bvp)
        # bvp = bvp_att*bvp + bvp
        bvp_4 = rearrange(bvp_3, 'b c w (l h) -> b c l w h', l=4, h=4)

        ptt_1 = self.ptt_seq_stem(x)
        ptt_2 = self.ptt_seq_max_1(ptt_1)
        # ver1
        ptt_3 = self.sa_bvp(ptt_2)
        # ver2
        # ptt_att = self.sa_bvp(ptt)
        # ptt = ptt_att*ptt + ptt
        ptt_4 = rearrange(ptt_3, 'b c h (l w) -> b c l w h', l=4, w=4)

        # att = ptt_4@bvp_4
        att = F.interpolate(ptt_4, scale_factor=(1, 1, 1 / 16))
        main_6 = main_5 * F.interpolate(att, scale_factor=(8, 1, 1)) + main_5

        main_7 = rearrange(main_6, 'b c l w h -> b c l (w h)')
        out_1 = self.max_vit(main_7)

        out_2 = torch.squeeze(out_1)
        out_3 = torch.mean(out_2, dim=-1)

        out_att = self.be_conv1d(out_3)
        out_4 = (1 + self.sigmoid(out_att)) * out_3
        out_5 = self.out_conv1d(out_4)
        out = torch.squeeze(out_5)
        # out = self.linear(out)
        return out
        # ,[main_1,main_2,main_3,main_4,main_5,main_6,main_7],\
        #    [bvp_1,bvp_2,bvp_3,bvp_4],[ptt_1,ptt_2,ptt_3,ptt_4],[att,out_att],\
        #    [out_1,out_2,out_3,out_4,out_5]
