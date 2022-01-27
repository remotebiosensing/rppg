import torch
from torch.nn import Module

from nets.blocks.attentionBlocks import AttentionBlock
from nets.blocks.blocks import EncoderBlock, DecoderBlock
from nets.modules.modules import DAModule


class AppearanceModel_2D(Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        # Appearance model
        super().__init__()
        self.a_conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       stride=1, padding=1)
        self.a_batch_Normalization1 = torch.nn.BatchNorm2d(out_channels)
        self.a_conv2 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1)
        self.a_batch_Normalization2 = torch.nn.BatchNorm2d(out_channels)
        self.a_dropout1 = torch.nn.Dropout2d(p=0.50)
        # Attention mask1
        self.attention_mask1 = AttentionBlock(out_channels)
        self.a_avg1 = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.a_conv3 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=3, stride=1,
                                       padding=1)
        self.a_Batch_Normalization3 = torch.nn.BatchNorm2d(out_channels * 2)
        self.a_conv4 = torch.nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels * 2, kernel_size=3,
                                       stride=1)
        self.a_Batch_Normalization4 = torch.nn.BatchNorm2d(out_channels * 2)
        self.a_dropout2 = torch.nn.Dropout2d(p=0.50)
        # Attention mask2
        self.attention_mask2 = AttentionBlock(out_channels * 2)

    def forward(self, inputs):
        # Convolution layer
        A1 = torch.tanh(self.a_batch_Normalization1(self.a_conv1(inputs)))
        A2 = torch.tanh(self.a_batch_Normalization2(self.a_conv2(A1)))
        A3 = self.a_dropout1(A2)
        # Calculate Mask1
        M1 = self.attention_mask1(A3)
        # Pooling
        A4 = self.a_avg1(A3)
        # Convolution layer
        A5 = torch.tanh(self.a_Batch_Normalization3(self.a_conv3(A4)))
        A6 = torch.tanh(self.a_Batch_Normalization4(self.a_conv4(A5)))
        A7 = self.a_dropout2(A6)
        # Calculate Mask2
        M2 = self.attention_mask2(A7)

        return M1, M2


class AppearanceModel_DA(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        # in_channels = 3
        # out_channels = 32
        super(AppearanceModel_DA, self).__init__()
        self.encoder_1 = EncoderBlock(in_channel=in_channels, out_channel=out_channels)  # conv3,conv3,maxpool2
        self.encoder_2 = EncoderBlock(in_channel=out_channels, out_channel=out_channels * 2)
        self.encoder_3 = EncoderBlock(in_channel=out_channels * 2, out_channel=out_channels * 4)  # conv3,conv3,maxpool2

        self.decoder_1 = DecoderBlock(in_channel=out_channels * 2, out_channel=out_channels, scale_facotr=2)
        self.decoder_2 = DecoderBlock(in_channel=out_channels * 4, out_channel=out_channels * 2, scale_facotr=2.25)

        self.damodul_1 = DAModule(out_channels)
        self.damodul_2 = DAModule(out_channels * 2)

        self.conv1x1_1 = torch.nn.Conv2d(out_channels // 4, 1, kernel_size=1, stride=1, padding=0)
        self.conv1x1_2 = torch.nn.Conv2d(out_channels // 2, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out_en_1 = self.encoder_1(x)
        out_en_2 = self.encoder_2(out_en_1)
        out_en_3 = self.encoder_3(out_en_2)

        out_de_1 = self.decoder_1(out_en_2)
        out_de_2 = self.decoder_2(out_en_3)

        out_concat_1 = out_de_1 + out_en_1
        out_concat_2 = out_de_2 + out_en_2

        out_da_1 = self.damodul_1(out_concat_1)
        M1 = self.conv1x1_1(out_da_1)

        out_da_2 = self.damodul_2(out_concat_2)
        M2 = self.conv1x1_2(out_da_2)

        return M1, M2
