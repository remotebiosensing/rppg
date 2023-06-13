import torch
device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
class BigSmall(torch.nn.Module):
    def __init__(self):
        super(BigSmall, self).__init__()
        self.big_branch = BigBranch()
        self.small_branch = SmallBranch()
        self.hr_model = LinearModel()

    def forward(self,inputs):
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
        pool_size = [2,2,4]
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=filters[0],kernel_size=3,padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=filters[0], out_channels=filters[1],kernel_size=3,padding=1)
        self.avg_pool1 = torch.nn.AvgPool2d(pool_size[0])
        self.conv3 = torch.nn.Conv2d(in_channels=filters[1], out_channels=filters[2],kernel_size=3,padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=filters[2], out_channels=filters[3],kernel_size=3,padding=1)
        self.avg_pool2 = torch.nn.AvgPool2d(pool_size[1])
        self.conv5 = torch.nn.Conv2d(in_channels=filters[3], out_channels=filters[4],kernel_size=3,padding=1)
        self.conv6 = torch.nn.Conv2d(in_channels=filters[4], out_channels=filters[5],kernel_size=3,padding=1)
        self.avg_pool3 = torch.nn.AvgPool2d(pool_size[2])

        self.drop_1 = torch.nn.Dropout2d(0.25)
        self.drop_2 = torch.nn.Dropout2d(0.5)

        self.relu = torch.nn.ReLU()

    def forward(self,x,n_frame=3):
        B, C, H, W = x.shape
        x = x.view(-1, n_frame, C, H, W)
        x = x[:,0,:,:,:]
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
        x = torch.reshape(x,[-1,C,H,W])
        return x

class SmallBranch(torch.nn.Module):
    def __init__(self):
        super(SmallBranch, self).__init__()
        filters = [32, 32, 32, 64]
        self.TSM1 = TSM_Block(3,filters[0],3,1)
        self.TSM2 = TSM_Block(filters[0],filters[1],3,1)
        self.TSM3 = TSM_Block(filters[1],filters[2],3,1)
        self.TSM4 = TSM_Block(filters[2],filters[3],3,1)



    def forward(self,x):
        x = self.TSM1(x,3,3)
        x = self.TSM2(x,3,3)
        x = self.TSM3(x,3,3)
        x = self.TSM4(x,3,3)
        return x

class TSM(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, input, n_frame=3, fold_div=3):
        B, C, H, W = input.shape
        # input = input.view(-1, n_frame, H, W, C)
        input = input.view(-1, n_frame, C, H, W)
        fold = C // fold_div
        last_fold = C - (fold_div - 1) * fold
        out1, out2, out3 = torch.split(input, [fold, fold, last_fold], 2)

        out1_pad, out1 = torch.split(out1, [1, n_frame - 1], 1)
        out1 = torch.cat((out1, out1_pad), 1)

        out2, out2_pad = torch.split(out2, [n_frame - 1, 1], 1)
        out2 = torch.cat((out2_pad, out2), 1)

        out = torch.cat([out1, out2, out3], 2)
        out = out.view([-1, C, H, W])

        return out


class TSM_Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.tsm1 = TSM()
        self.t_conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,padding=padding)
        self.relu = torch.nn.ReLU()

    def forward(self, input, n_frame=3, fold_div=3):
        t = self.tsm1(input, n_frame, fold_div)
        t = self.relu(self.t_conv1(t))
        return t


