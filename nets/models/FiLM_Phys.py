import torch
import torch.nn as nn

from nets.blocks.decoder_blocks import decoder_block
from nets.blocks.encoder_blocks import encoder_block
from nets.blocks.blocks import ConvBlock3D

import numpy as np

def conv(ic, oc, k, s, p):
    return nn.Sequential(
        nn.Conv2d(ic, oc, k, s, p),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(oc),
    )


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.model = nn.Sequential(
            # conv(3, 128, 5, 1, 2),
            # conv(128, 128, 3, 1, 1),
            # conv(128, 128, 4, 2, 1),
            # conv(128, 128, 4, 2, 1),
            # conv(128, 128, 4, 2, 1),
            conv(3, 128, 5, 2, 2),
            conv(128, 128, 3, 2, 1),
            conv(128, 128, 3, 2, 1),
            conv(128, 128, 3, 1, 1),
            conv(128, 128, 3, 1, 1),
            # conv(3, 128, 4, 2, 2),
            # conv(128, 128, 4, 2, 1),
            # conv(128, 128, 4, 2, 1),
            # conv(128, 128, 4, 2, 1),
        )

    def forward(self, x):
        return self.model(x)


class PhysNetFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super(PhysNetFeatureExtractor, self).__init__()
        self.physnetfe = torch.nn.Sequential(
            encoder_block(),
            decoder_block()
        )

    def forward(self, x):
        [batch, channel, length, width, height] = x.shape
        #return self.physnet(x).view(-1, length)
        return self.physnetfe(x)


class FiLMBlock(nn.Module):
    def __init__(self):
        super(FiLMBlock, self).__init__()

    def forward(self, x, gamma, beta):
        beta = beta.view(x.size(0), x.size(1), 1, 1, 1)
        gamma = gamma.view(x.size(0), x.size(1), 1, 1, 1)

        x = gamma * x + beta

        return x


class ResBlock(nn.Module):
    def __init__(self, in_place, out_place):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_place, out_place, 1, 1, 0)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_place, out_place, 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(out_place)
        self.film = FiLMBlock()
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, beta, gamma):
        x = self.conv1(x)
        x = self.relu1(x)
        identity = x

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.film(x, beta, gamma)
        x = self.relu2(x)

        x = x + identity

        return x

class eBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(eBlock, self).__init__()

        self.conv1 = nn.Conv3d(in_channel, out_channel, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        return x

class dBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(dBlock, self).__init__()

        self.conv1 = nn.ConvTranspose3d(in_channel, out_channel, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.relu1 = nn.ELU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        return x

class feBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(feBlock, self).__init__()

        self.conv1 = nn.Conv3d(in_channel, out_channel, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.relu1 = nn.ReLU(inplace=True)
        self.film = FiLMBlock()

    def forward(self, x, beta, gamma):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.film(x, beta, gamma)
        x = self.relu1(x)

        return x

class fdBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(fdBlock, self).__init__()

        self.conv1 = nn.ConvTranspose3d(in_channel, out_channel, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.relu1 = nn.ELU(inplace=True)
        self.film = FiLMBlock()

    def forward(self, x, beta, gamma):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.film(x, beta, gamma)
        x = self.relu1(x)

        return x

class PhysNetClassifier(nn.Module):
    def __init__(self):
        super(PhysNetClassifier, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.AdaptiveMaxPool3d((32, 1, 1)),  # spatial adaptive pooling
            torch.nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0))

    def forward(self, x):
        x = self.model(x)
        [batch, channel, length, width, height] = x.shape
        return x.view(-1, length)

# Create CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv_layer1 = self._conv_layer_set(3, 32)
        self.conv_layer2 = self._conv_layer_set(32, 64)
        self.fc1 = nn.Linear(345600, 1024)
        self.relu = nn.LeakyReLU()

    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer

    def forward(self, x):
        # Set 1
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)

        return out

class FiLM(nn.Module):
    def __init__(self):
        super(FiLM, self).__init__()

        dim_question = 11
        # Linear에서 나온 결과의 절반은 beta, 절반은 gamma
        # beta, gamma 모두 ResBlock 하나당 n_channels개씩 feed
        #self.film_generator = nn.Linear(dim_question, 2 * n_res_blocks * n_channels)
        self.film_generator = CNNModel()

        self.fc1 = nn.Linear(1024, 16 * 2)
        self.fc2 = nn.Linear(1024, 32 * 2)
        self.fc3 = nn.Linear(1024, 64 * 2)

        self.film = FiLMBlock()

        #        self.feature_extractor = FeatureExtractor()

        #self.res_blocks = nn.ModuleList()
#        for _ in range(n_res_blocks):
#            self.res_blocks.append(ResBlock(n_channels + 2, n_channels))

#        self.n_res_blocks = n_res_blocks
        #self.n_channels = n_channels

        self.blocks = nn.ModuleList()
        self.blocks.append(feBlock(3, 16, [1, 5, 5], [1, 1, 1], [0, 2, 2]))
        self.blocks.append(nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)))
        self.blocks.append(eBlock(16, 32, [3, 3, 3], [1, 1, 1], [1, 1, 1]))
        self.blocks.append(eBlock(32, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1]))
        self.blocks.append(nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)))
        self.blocks.append(eBlock(64, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1]))
        self.blocks.append(eBlock(64, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1]))
        self.blocks.append(nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)))
        self.blocks.append(eBlock(64, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1]))
        self.blocks.append(eBlock(64, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1]))
        self.blocks.append(nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)))
        self.blocks.append(eBlock(64, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1]))
        self.blocks.append(eBlock(64, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1]))

        self.blocks.append(dBlock(64, 64, [4, 1, 1], [2, 1, 1], [1, 0, 0]))
        self.blocks.append(fdBlock(64, 64, [4, 1, 1], [2, 1, 1], [1, 0, 0]))

        self.classifier = PhysNetClassifier()

    def forward(self, x): #(image, question)
        batch_size = x.size(0)

        #x_feature = self.feature_extractor(x)
       # film_vector, n_channels = self.film_generator(x)
       # film_vector = film_vector.view(batch_size, 2, n_channels)
            #batch_size, self.n_res_blocks, 2, self.n_channels)


        '''
        d = x.size(2)
        coordinate = torch.arange(-1, 1 + 0.00001, 2 / (d - 1)).cuda()
        coordinate_x = coordinate.expand(batch_size, 1, d, d)
        coordinate_y = coordinate.view(d, 1).expand(batch_size, 1, d, d)

        for i, res_block in enumerate(self.res_blocks):
            beta = film_vector[:, i, 0, :]
            gamma = film_vector[:, i, 1, :]

            x = torch.cat([x, coordinate_x, coordinate_y], 1)
            x = res_block(x, beta, gamma)
        '''
        film_vector = self.film_generator(x)
        for i, block in enumerate(self.blocks):
            '''
            if i in [0, 2, 3, 5, 6, 8, 9, 11, 12, 13, 14]:

                if i == 0:
                    n_channels = 16
                    filmv = self.fc1(film_vector)
                elif i == 2:
                    n_channels = 32
                    filmv = self.fc2(film_vector)
                else:
                    n_channels =64
                    filmv = self.fc3(film_vector)

                filmv = filmv.view(batch_size, 2, n_channels)

                beta = filmv[:, 0, :]
                gamma = filmv[:, 1, :]
                x = block(x, beta, gamma)

            '''

            if i == 0:
                n_channels = 16
                filmv = self.fc1(film_vector)
                filmv = filmv.view(batch_size, 2, n_channels)

                beta = filmv[:, 0, :]
                gamma = filmv[:, 1, :]
                x = block(x, beta, gamma)
            elif i == 14:
                n_channels = 64
                filmv = self.fc3(film_vector)
                filmv = filmv.view(batch_size, 2, n_channels)

                beta = filmv[:, 0, :]
                gamma = filmv[:, 1, :]
                x = block(x, beta, gamma)

            else:
                x= block(x)
        # feature = x
        x = self.classifier(x)

        return x  # , feature


def make_model(model_dict):
    return FiLM(model_dict['n_res_blocks'], model_dict['n_classes'], model_dict['n_channels'])

if __name__ =='__main__':
    FiLM()