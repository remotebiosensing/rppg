import torch
import torch.nn as nn

from tqdm import tqdm
import wandb
import json

with open('/home/paperc/PycharmProjects/BPNET/parameter.json') as f:
    json_data = json.load(f)
    param = json_data.get("parameters")
    sampling_rate = json_data.get("parameters").get("sampling_rate")


'''
nn.Conv1d expects as 3-dimensional input in the shape of [batch_size, channels, seq_len]
'''


class bvp2abp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(bvp2abp, self).__init__()
        self.in_channel = in_channels
        # 1D Convolution 3 size kernel (1@7500 -> 32@7500)
        self.conv_layer1 = torch.nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=1,
                      padding="same", padding_mode="replicate"),
            nn.BatchNorm1d(out_channels)
        )

        # Convolution 3x3 kernel Layer 2 (32@7500 -> 32@7500)
        self.conv_layer2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                            kernel_size=kernel_size, stride=1,
                            padding="same", padding_mode="replicate"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self.conv_layer3 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=out_channels, out_channels=int(out_channels / 2),
                            kernel_size=kernel_size*2, stride=1,
                            padding="same", padding_mode="replicate"),
            nn.BatchNorm1d(int(out_channels / 2)),
        )
        self.conv_layer4 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=int(out_channels / 2), out_channels=int(out_channels / 4),
                            kernel_size=kernel_size*2, stride=1,
                            padding="same", padding_mode="replicate"),
            nn.BatchNorm1d(int(out_channels / 4)),
            nn.ReLU()
        )

        self.conv_layer5 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=int(out_channels / 4), out_channels=1,
                            kernel_size=kernel_size*4, stride=1,
                            padding="same", padding_mode="replicate"),
            nn.BatchNorm1d(1)
        )
        self.conv_layer6 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=1,
                            kernel_size=kernel_size*4, stride=1,
                            padding="same", padding_mode="replicate")
        )

        # Dropout
        self.a_dropout = torch.nn.Dropout1d(p=0.5)
        # Average-pooling 2x2 kernel Layer 3 (32@7500 -> 32@3750)
        self.avg_pool1 = torch.nn.AvgPool1d(kernel_size=2, stride=2)

        self.dense1 = torch.nn.Linear(int(((param["chunk_size"] / 125) * sampling_rate["60"])),
                                      int((((param["chunk_size"] / 125) * sampling_rate["60"]) / 4) * 3))
        self.dense2 = torch.nn.Linear(int((((param["chunk_size"] / 125) * sampling_rate["60"]) / 4) * 3),
                                      int((param["chunk_size"] / 125) * sampling_rate["60"]))
        self.sigmoid = torch.nn.Sigmoid()

        self.att = torch.nn.MultiheadAttention(embed_dim=3600, num_heads=360, batch_first=True, dropout=0.2)

        self.relu = torch.nn.ReLU(inplace=True)

    # TODO FORWARD안에 FEATURE 앞에다가 DATALOADER(__GETITEM__())에서 얻은 크기 정보 추가
    def forward(self, ple_input):
        # print('shape of ple_input :', ple_input.shape)
        ple_input = torch.reshape(ple_input, (-1, self.in_channel, int((param["chunk_size"] / 125) * 60)))  # [ batch , channel, size]
        # print('reshape of ple_input :', ple_input.shape)
        c1 = self.conv_layer1(ple_input)
        # print('shape of conv1 output :', c1.shape)
        c2 = self.conv_layer2(c1)
        # print('shape of conv2 output :', c2.shape)
        d1 = self.a_dropout(c2)
        # print('shape of dropout1 output :', d1.shape)
        c3 = self.conv_layer3(d1)
        # print('shape of conv3 output :', c3.shape)
        c4 = self.conv_layer4(c3)
        # print('shape of conv4 output :', c4.shape)
        d2 = self.a_dropout(c4)
        # print('shape of dropout2 output :', d2.shape)
        c5 = self.conv_layer5(d2)
        # print('shape of conv5 output :', c5.shape)
        c6 = self.conv_layer6(c5)
        # print('shape of conv6 output :', c6.shape)
        feature2 = self.sigmoid(c6) + c5
        # att, _ = self.att(feature2, feature2, feature2)
        # print('shape of feature output :', feature2.shape)
        # p1 = self.avg_pool1(feature2)
        # print('shape of avg_pool1 output :', p1.shape)
        out = self.dense1(feature2)
        # print('shape of dense1 output :', out.shape)
        abp_pred = self.dense2(out)
        # print('shape of dense2 output :', out.shape)
        return abp_pred



def train(model, loader, loss1, loss2, optimizer, epoch, batch, config):
    costarr = []
    wandb.watch(models=model, criterion=loss1, log="all", log_freq=10)

    for epoch in tqdm(range(epoch)):
        avg_cost = 0

        for X, Y in loader:
            hypothesis = model(X)

            optimizer.zero_grad()
            cost1 = loss1(hypothesis, Y)
            cost2 = loss2(hypothesis, Y)
            cost = cost1 * cost2
            cost.backward()
            optimizer.step()

            avg_cost += cost / batch
        costarr.append(avg_cost.__float__())
        print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))
