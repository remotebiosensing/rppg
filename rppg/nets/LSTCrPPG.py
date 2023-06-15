import torch

class LSTCrPPG(torch.nn.Module):
    def __init__(self, frames=32):
        super(LSTCrPPG, self).__init__()
        self.encoder_block = EncoderBlock()
        self.decoder_block = DecoderBlock()

    def forward(self, x):
        e = self.encoder_block(x)
        out = self.decoder_block(e)
        return out.squeeze()


class EncoderBlock(torch.nn.Module):
    def __init__(self):
        super(EncoderBlock, self).__init__()
        #, in_channel, out_channel, kernel_size, stride, padding
        self.encoder_block1 = torch.nn.Sequential(
            ConvBlock3D(3, 16, [3,3,3], [1,1,1], [1,1,1]),
            ConvBlock3D(16, 16, [3,3,3], [1,1,1], [1,1,1]),
            torch.nn.BatchNorm3d(16)
        )
        self.encoder_block2 = torch.nn.Sequential(
            torch.nn.AvgPool3d(2),
            ConvBlock3D(16, 16, [3, 3, 3], [1, 1, 1], [1,1,1]),
            ConvBlock3D(16, 16, [3, 3, 3], [1, 1, 1], [1,1,1]),
            torch.nn.BatchNorm3d(16)
        )
        self.encoder_block3 = torch.nn.Sequential(
            torch.nn.AvgPool3d(2),
            ConvBlock3D(16, 32, [3, 3, 3], [1, 1, 1], [1,1,1]),
            ConvBlock3D(32, 32, [3, 3, 3], [1, 1, 1], [1,1,1]),
            torch.nn.BatchNorm3d(32)
        )
        self.encoder_block4 = torch.nn.Sequential(
            torch.nn.AvgPool3d(2),
            ConvBlock3D(32, 32, [3, 3, 3], [1, 1, 1], [1,1,1]),
            ConvBlock3D(32, 32, [3, 3, 3], [1, 1, 1], [1,1,1]),
            torch.nn.BatchNorm3d(32)
        )
        self.encoder_block5 = torch.nn.Sequential(
            torch.nn.AvgPool3d(2),
            ConvBlock3D(32, 64, [3, 3, 3], [1, 1, 1], [1,1,1]),
            ConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1,1,1]),
            torch.nn.BatchNorm3d(64)
        )
        self.encoder_block6 = torch.nn.Sequential(
            torch.nn.AvgPool3d(2),
            ConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1,1,1]),
            torch.nn.BatchNorm3d(64)
        )
        self.encoder_block7 = torch.nn.Sequential(
            ConvBlock3D(64, 64, [5, 3, 3], [1, 1, 1], [0,1,1]),
            torch.nn.BatchNorm3d(64)
        )

    def forward(self, x):
        e1 = self.encoder_block1(x)
        e2 = self.encoder_block2(e1)
        e3 = self.encoder_block3(e2)
        e4 = self.encoder_block4(e3)
        e5 = self.encoder_block5(e4)
        e6 = self.encoder_block6(e5)
        e7 = self.encoder_block7(e6)

        return [e7,e6,e5,e4,e3,e2,e1]

class DecoderBlock(torch.nn.Module):
    def __init__(self):
        super(DecoderBlock, self).__init__()
        self.decoder_block6_transpose = torch.nn.ConvTranspose3d(64,64,[5,1,1],[1,1,1])
        self.decoder_block6 = torch.nn.Sequential(
            ConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1,1,1]),
            torch.nn.BatchNorm3d(64)
        )
        self.decoder_block5_transpose =torch.nn.ConvTranspose3d(64, 64, [4, 1, 1],[2,1,1])
        self.decoder_block5 = torch.nn.Sequential(
            ConvBlock3D(64, 32, [3, 3, 3], [1, 1, 1],[1,1,1]),
            ConvBlock3D(32, 32, [3, 3, 3], [1, 1, 1],[0,1,1]),
            torch.nn.BatchNorm3d(32)
        )
        self.decoder_block4_transpose = torch.nn.ConvTranspose3d(32, 32, [4, 1, 1],[2,1,1])
        self.decoder_block4 = torch.nn.Sequential(
            ConvBlock3D(32, 32, [3, 3, 3], [1, 1, 1], [1,1,1]),
            ConvBlock3D(32, 32, [3, 3, 3], [1, 1, 1], [0,1,1]),
            torch.nn.BatchNorm3d(32)
        )
        self.decoder_block3_transpose = torch.nn.ConvTranspose3d(32, 32, [4, 1, 1],[2,1,1])
        self.decoder_block3 = torch.nn.Sequential(
            ConvBlock3D(32, 16, [3, 3, 3], [1, 1, 1], [1,1,1]),
            ConvBlock3D(16, 16, [3, 3, 3], [1, 1, 1], [0,1,1]),
            torch.nn.BatchNorm3d(16)
        )
        self.decoder_block2_transpose = torch.nn.ConvTranspose3d(16, 16, [4, 1, 1],[2,1,1])
        self.decoder_block2 = torch.nn.Sequential(
            ConvBlock3D(16, 16, [3, 3, 3], [1, 1, 1], [1,1,1]),
            ConvBlock3D(16, 16, [3, 3, 3], [1, 1, 1], [0,1,1]),
            torch.nn.BatchNorm3d(16)
        )
        self.decoder_block1_transpose = torch.nn.ConvTranspose3d(16, 16, [4, 1, 1],[2,1,1])
        self.decoder_block1 = torch.nn.Sequential(
            ConvBlock3D(16, 3, [3, 3, 3], [1, 1, 1], [1,1,1]),
            ConvBlock3D(3, 3, [3, 3, 3], [1, 1, 1], [0,1,1]),
            torch.nn.BatchNorm3d(3)
        )
        self.predictor = torch.nn.Conv3d(3, 1 ,[1,4,4])



    def forward(self, encoded_feature):
        d6 = self.decoder_block6(self.TARM(encoded_feature[1], self.decoder_block6_transpose(encoded_feature[0])))
        d5 = self.decoder_block5(self.TARM(encoded_feature[2],self.decoder_block5_transpose(d6)))
        d4 = self.decoder_block4(self.TARM(encoded_feature[3],self.decoder_block4_transpose(d5)))
        d3 = self.decoder_block3(self.TARM(encoded_feature[4],self.decoder_block3_transpose(d4)))
        d2 = self.decoder_block2(self.TARM(encoded_feature[5],self.decoder_block2_transpose(d3)))
        d1 = self.decoder_block1(self.TARM(encoded_feature[6],self.decoder_block1_transpose(d2)))
        predictor = self.predictor(d1)
        return predictor

    def TARM(self, e,d):
        target = d
        shape = d.shape
        e = torch.nn.functional.adaptive_avg_pool3d(e,d.shape[2:])
        e = e.view(e.shape[0],e.shape[1], shape[2],-1)
        d = d.view(d.shape[0], shape[1], shape[2], -1)
        temporal_attention_map = e @ torch.transpose(d,3,2)
        temporal_attention_map = torch.nn.functional.softmax(temporal_attention_map,dim=-1)
        refined_map = temporal_attention_map@e
        out = refined_map#( 1 + torch.reshape(refined_map,shape)) * target
        return out

class ConvBlock3D(torch.nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(ConvBlock3D, self).__init__()
        self.conv_block_3d = torch.nn.Sequential(
            torch.nn.Conv3d(in_channel, out_channel, kernel_size, stride, padding),
            torch.nn.ELU()
        )

    def forward(self, x):
        return self.conv_block_3d(x)


