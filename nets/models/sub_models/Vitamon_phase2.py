import torch
import torch.nn as nn

class BasicConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class Vitamon_phase2(nn.Module):
    def __init__(self):
        super(Vitamon_phase2, self).__init__()

        self.conv2d1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=7, out_channels=32, kernel_size=(3,3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv2d2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv2d3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.avg_pool = torch.nn.AvgPool2d((4,4))
        self.avg_pool_2 = torch.nn.AvgPool2d((3,3))

        self.dropout = torch.nn.Dropout(0.5)

        self.conv2d4 = torch.nn.Conv2d(in_channels=64,out_channels=25, kernel_size=(3,3))

        self.flatten = torch.nn.Flatten()

        self.ful_conn1 = torch.nn.Linear(in_features=25, out_features=25)

        self.ful_conn2 = torch.nn.Linear(in_features=25, out_features=25)

    def forward(self, x):
        c1 = self.conv2d1(x) # input : (10,7,224,224), output : (10,32,222,222)
        c2 = self.conv2d2(c1) # input : (10,32,222,222), output : (10,32,220,220)
        c3 = self.conv2d3(c2) # input : (10,32,220,220), output : (10,64,218,218)
        c4 = self.conv2d4(c3) # input : (10,64,218,218), output : (10,25,216,216)
        p1 = self.avg_pool(c4) # input : (10,25,216,216), output : (10,64,54,54)
        p2 = self.avg_pool(p1) # input : (10,25,54,54), output : (10,25,13,13)
        p3 = self.avg_pool(p2) # input : (10,25,13,13), output : (10,25,3,3)
        p4 = self.avg_pool_2(p3) # input : (10,25,3,3), output : (10,25,1,1)
        d1 = self.dropout(p4) # input : (10,25,1,1), output : (10,25,1,1)

        f1 = self.flatten(d1) # input : (10,25,1,1), output : (10,25)
        fu1 = self.ful_conn1(f1) # input : (10,25), output : (10,25)
        fu2 = self.ful_conn2(fu1) # input : (10,25), output : (10,25)

        return fu2

    def summary(self):
        self.model.summary()

