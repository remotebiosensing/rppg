import torch
from rppg.nets.TSCAN import TSM


class BigSmall(torch.nn.Module):
    def __init__(self, time_length=3):
        super(BigSmall, self).__init__()
        self.time_length = time_length
        self.big_branch = BigBranch()
        self.small_branch = SmallBranch(self.time_length)
        self.hr_model = LinearModel()

    def forward(self, inputs):
        big_out = self.big_branch(inputs[0])
        small_out = self.small_branch(inputs[1])
        sum = big_out + small_out
        out = self.hr_model(sum)
        return out


class LinearModel(torch.nn.Module):
    def __init__(self, in_channel=5184):
        super().__init__()
        self.f_linear1 = torch.nn.Linear(in_channel, 128, bias=True)
        self.f_linear2 = torch.nn.Linear(128, 1, bias=True)

    def forward(self, input):
        f1 = torch.flatten(input, start_dim=1)
        f2 = torch.nn.functional.relu(self.f_linear1(f1))
        f3 = self.f_linear2(f2)
        return f3


class BigBranch(torch.nn.Module):
    def __init__(self):
        super(BigBranch, self).__init__()
        filters = [32, 32, 32, 64, 64, 64]
        pool_size = [2, 2, 4]
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=filters[0], kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=filters[0], out_channels=filters[1], kernel_size=3, padding=1)
        self.avg_pool1 = torch.nn.AvgPool2d(pool_size[0])
        self.conv3 = torch.nn.Conv2d(in_channels=filters[1], out_channels=filters[2], kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=filters[2], out_channels=filters[3], kernel_size=3, padding=1)
        self.avg_pool2 = torch.nn.AvgPool2d(pool_size[1])
        self.conv5 = torch.nn.Conv2d(in_channels=filters[3], out_channels=filters[4], kernel_size=3, padding=1)
        self.conv6 = torch.nn.Conv2d(in_channels=filters[4], out_channels=filters[5], kernel_size=3, padding=1)
        self.avg_pool3 = torch.nn.AvgPool2d(pool_size[2])

        self.drop_1 = torch.nn.Dropout2d(0.25)
        self.drop_2 = torch.nn.Dropout2d(0.5)

        self.relu = torch.nn.ReLU()

    def forward(self, x, n_frame=3):
        B, C, H, W = x.shape
        x = x.view(-1, n_frame, C, H, W)
        x = x[:, 0, :, :, :]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.avg_pool1(x)
        x = self.drop_1(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.avg_pool2(x)
        x = self.drop_2(x)
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.avg_pool3(x)
        x = self.drop_2(x)
        B, C, H, W = x.shape
        x = x.unsqueeze(1).expand(B, n_frame, C, H, W)
        x = torch.reshape(x, [-1, C, H, W])
        return x


class SmallBranch(torch.nn.Module):
    def __init__(self, time_length=3):
        super(SmallBranch, self).__init__()
        self.time_length = time_length
        filters = [32, 32, 32, 64]
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=filters[0], kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=filters[0], out_channels=filters[1], kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=filters[1], out_channels=filters[2], kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=filters[2], out_channels=filters[3], kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU()
        self.tsm1 = WTSM(time_length=self.time_length, fold_div=3)
        self.tsm2 = WTSM(time_length=self.time_length, fold_div=3)
        self.tsm3 = WTSM(time_length=self.time_length, fold_div=3)
        self.tsm4 = WTSM(time_length=self.time_length, fold_div=3)

    def forward(self, x):
        s1 = self.relu(self.conv1(self.tsm1(x)))
        s2 = self.relu(self.conv2(self.tsm2(s1)))
        s3 = self.relu(self.conv3(self.tsm3(s2)))
        s4 = self.relu(self.conv4(self.tsm4(s3)))

        return s4


class WTSM(TSM):
    def forward(self, input):
        B, C, H, W = input.shape
        input = input.view(-1, self.time_length, C, H, W)

        fold = C // self.fold_div
        last_fold = C - (self.fold_div - 1) * fold

        out1, out2, out3 = torch.split(input, [fold, fold, last_fold], dim=2)

        up_out1 = torch.cat((out1[:, -1:, :, :, :], out1[:, :-1, :, :, :]), dim=1)
        donw_out2 = torch.cat((out2[:, 1:, :, :, :], out2[:, :1, :, :, :]), dim=1)
        wtsm_out = torch.cat((up_out1, donw_out2, out3), dim=2).view(B, C, H, W)

        return wtsm_out
