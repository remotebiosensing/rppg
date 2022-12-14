import torch
import torch.nn as nn
import torchvision.models.inception as inception
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
class InceptionA(nn.Module):

    def __init__(self, input_channels, pool_features):
        super().__init__()
        self.branch1x1 = BasicConv2d(input_channels, 64, kernel_size=1)

        self.branch5x5 = nn.Sequential(
            BasicConv2d(input_channels, 48, kernel_size=1),
            BasicConv2d(48, 64, kernel_size=5, padding=2)
        )

        self.branch3x3 = nn.Sequential(
            BasicConv2d(input_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1),
            BasicConv2d(96, 96, kernel_size=3, padding=1)
        )

        self.branchpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(input_channels, pool_features, kernel_size=3, padding=1)
        )

    def forward(self, x):

        #x -> 1x1(same)
        branch1x1 = self.branch1x1(x)

        #x -> 1x1 -> 5x5(same)
        branch5x5 = self.branch5x5(x)
        #branch5x5 = self.branch5x5_2(branch5x5)

        #x -> 1x1 -> 3x3 -> 3x3(same)
        branch3x3 = self.branch3x3(x)

        #x -> pool -> 1x1(same)
        branchpool = self.branchpool(x)

        outputs = [branch1x1, branch5x5, branch3x3, branchpool]

        return torch.cat(outputs, 1)

class Vitamon(nn.Module):
    def __init__(self):
        super(Vitamon, self).__init__()

        self.conv2d1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=25, out_channels=32, kernel_size=(3,3)),
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
        self.max_pool = torch.nn.MaxPool2d((2,2))

        self.conv2d4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=80, kernel_size=(3,3)),
            nn.BatchNorm2d(80),
            nn.ReLU(),
        )
        self.conv2d5 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=80, out_channels=192, kernel_size=(3,3)),
            nn.BatchNorm2d(192),
            nn.ReLU(),
        )

        # define the inception block
        self.inception = InceptionA(input_channels=192, pool_features=25)

        self.conv2d6 = torch.nn.Conv2d(in_channels=249, out_channels=25, kernel_size=(1,1))
        # out_channels size 조절 위해 사용

        self.avg_pool = torch.nn.AvgPool2d((3,3))

        self.dropout = torch.nn.Dropout(0.5)

        self.flatten = torch.nn.Flatten()

        self.ful_conn1 = torch.nn.Linear(in_features=25, out_features=25)

        self.ful_conn2 = torch.nn.Linear(in_features=25, out_features=25)

    def forward(self, x):
        x = self.conv2d1(x) #input : (10,25,224,224) output : (10,32,222,222)
        x = self.conv2d2(x) #input : (10,32,222,222) output : (10,32,220,220)
        x = self.conv2d3(x) #input : (10,32,220,220) output : (10,64,218,218)
        x = self.max_pool(x) #input : (10,64,218,218) output : (10,64,109,109)
        x = self.conv2d4(x) #input : (10,64,109,109) output : (10,80,107,107)
        x = self.conv2d5(x) #input : (10,80,107,107) output : (10,192,105,105)
        x = self.max_pool(x) #input : (10,192,105,105) output : (10,192,52,52)
        x = self.inception(x) #input : (10,192,52,52) output : (10,249,52,52)
        x = self.conv2d6(x) #input : (10,249,52,52) output : (10,25,52,52)
        x = self.avg_pool(x) #input : (10,25,52,52) output : (10,25,17,17)
        x = self.avg_pool(x) #input : (10,25,17,17) output : (10,25,5,5)
        x = self.avg_pool(x) #input : (10,25,5,5) output : (10,25,1,1)
        #x = self.avg_pool(x) #input : (4,25,6,6) output : (4,25,3,3)
        x = self.dropout(x) #input : (10,25,1,1) output : (10,25,1,1)
        x = self.flatten(x) #input : (10,25,1,1) output : (10,25)
        x = self.ful_conn1(x) #input : (10,25) output : (10,25)
        x = self.ful_conn2(x) #input : (10,25) output : (10,25)

        return x

    def summary(self):
        self.model.summary()