import torch
from torch import einsum
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from torchvision import transforms

time_length = 64

class APNET(torch.nn.Module):
    def __init__(self):
        super(APNET, self).__init__()

        self.f_m = APNET_Backbone()
        self.l_m = APNET_Backbone()
        self.r_m = APNET_Backbone()
        # Feature fusion 레이어
        self.feature_fusion = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=48, out_channels=16, kernel_size=1),
            torch.nn.ReLU()
        )

        # Attention 메커니즘을 위한 추가 레이어
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(3, 3),
            torch.nn.Softmax(dim=1)
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


class sub(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(sub, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        x = self.conv1(x)
        x = self.sigmoid(x)

        return x.unsqueeze(2).expand(-1, -1, time_length, -1)

class APNET_Backbone(torch.nn.Module):
    def __init__(self):
        super(APNET_Backbone, self).__init__()

        self.dog_transform = transforms.Compose([
            transforms.Lambda(lambda x: x.float() / 255.0),
            transforms.Lambda(lambda x: x[1] - x[0]),  # Gaussian 차이
            transforms.Lambda(lambda x: torch.abs(x)),  # 절대값 취하기
            transforms.Normalize([0.5], [0.5]),  # -1 ~ 1 사이의 값으로 정규화
        ])

        self.main_seq_stem = torch.nn.Sequential(
            Rearrange('b l c h w -> (b l) c h w'),
            torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
            Rearrange('(b l) c h w -> b c l (h w)', l=time_length)
        )
        # padding =  kernel /2
        # layerdim = dim * expand 대신 ㅅㅏ용
        self.main_seq_max_1 = torch.nn.Sequential(
            MaxViT_layer(layer_depth=1, layer_dim_in=16, layer_dim=32,
                         kernel=(2, 16), dilation=(1, 8), padding=(1,8),
                         mbconv_expansion_rate=4, mbconv_shrinkage_rate=0.5, w=4, dim_head=8, dropout=0.1, flag=False)
        )
        self.sa_main = SpatialAttention(in_channels=32)
        self.rnn = torch.nn.LSTM(input_size=64*32, hidden_size=32, num_layers=2, batch_first=True, bidirectional=True,dropout=0.2)
        self.adaptive = torch.nn.AdaptiveAvgPool2d((32, 16))
        self.max_vit = MaxViT_layer(layer_depth=2, layer_dim_in=1, layer_dim=32,
                                    kernel=3, dilation=1, padding=1,
                                    mbconv_expansion_rate=4, mbconv_shrinkage_rate=0.25, w=4, dim_head=32, dropout=0.1,
                                    flag=False)
        self.be_conv1d = torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding="same")
        self.out_conv1d = torch.nn.Conv1d(in_channels=32, out_channels=1, kernel_size=1)

        self.sigmoid = torch.nn.Sigmoid()

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

class SpatialAttention(torch.nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(SpatialAttention, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool2d(1)

        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels // reduction_ratio, kernel_size=1)
        self.conv2 = torch.nn.Conv2d(in_channels=in_channels // reduction_ratio, out_channels=in_channels, kernel_size=1)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        avg_out = self.conv2(self.conv1(self.avg_pool(x)))
        max_out = self.conv2(self.conv1(self.max_pool(x)))

        out = self.sigmoid(avg_out + max_out)
        return x * out

class MaxViT_layer(torch.nn.Module):
    def __init__(self,layer_depth,layer_dim_in, layer_dim,kernel,dilation,padding, mbconv_expansion_rate,mbconv_shrinkage_rate,w,dim_head,dropout,flag):
        super(MaxViT_layer, self).__init__()
        self.layers = torch.nn.ModuleList([])
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

class MaxViT_Block(torch.nn.Module):
    def __init__(self, stage_dim_in, layer_dim,kernel,dilation,padding, is_first, mbconv_expansion_rate,mbconv_shrinkage_rate,w,dim_head,dropout):
        super(MaxViT_Block, self).__init__()

        # stride = 2 if 0 else 1

        self.l0 = torch.nn.Sequential(
            torch.nn.Conv2d(stage_dim_in, layer_dim, 1),
            torch.nn.BatchNorm2d(layer_dim),
            torch.nn.SiLU(),

        )
        self.l1 = torch.nn.Sequential(
            torch.nn.Conv2d(layer_dim, layer_dim, kernel_size=kernel, stride=1, padding='same', dilation=dilation,groups=layer_dim),
            SqueezeExcitation(layer_dim, shrinkage_rate=mbconv_shrinkage_rate),
            torch.nn.Conv2d(layer_dim, layer_dim, 1),
            torch.nn.BatchNorm2d(layer_dim)
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


class SqueezeExcitation(torch.nn.Module):
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = torch.nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            torch.nn.Linear(dim, hidden_dim, bias = False),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, dim, bias = False),
            torch.nn.Sigmoid(),
            Rearrange('b c -> b c 1 1')
        )

    def forward(self, x):
        return x * self.gate(x)

class PreNormResidual(torch.nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class Attention(torch.nn.Module):
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

        self.to_qkv = torch.nn.Linear(dim, dim * 3, bias = False)

        self.attend = torch.nn.Sequential(
            torch.nn.Softmax(dim = -1),
            torch.nn.Dropout(dropout)
        )

        self.to_out = torch.nn.Sequential(
            torch.nn.Linear(dim, dim, bias = False),
            torch.nn.Dropout(dropout)
        )

        # relative positional bias

        self.rel_pos_bias = torch.nn.Embedding((2 * window_size - 1) ** 2, self.heads)

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

class FeedForward(torch.nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, inner_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(inner_dim, dim),
            torch.nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
