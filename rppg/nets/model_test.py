import torch

from rppg.nets.APNETv2 import APNETv2 as APNETv2
from rppg.nets.DeepPhys import DeepPhys as DeepPhys
from rppg.nets.ETArPPGNet import ETArPPGNet as ETArPPGNet
from rppg.nets.PhysNet import PhysNet as PhysNet

device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

model_name = "ETArPPGNet"

if __name__ == '__main__':

    if model_name == "PhysNet":
        img = torch.rand(32,3,32,32,32).to(device) # [batch, channel, length, width, height]
        net = PhysNet().to(device)
        out = net(img) #[batch, length]
    elif model_name == "DeepPhys":
        img = torch.rand(32,2,3,36,36).to(device) # [batch, norm + diff, channel, width, height]
        net = DeepPhys().to(device)
        out = net(img) # [batch, 1]
    elif model_name == "ETArPPGNet":
        img = torch.rand(32,30,3, 10, 224, 224).to(device) #[batch, block, Channel, time, width, height]
        net = ETArPPGNet().to(device)
        out = net(img) #[batch, block * time]
    elif model_name == "APNETv2":
        img = torch.rand(3,32, 64,3,30,30).to(device) #[fore head + left cheek + right cheek, batch, time, channel, width, height]
        net = APNETv2().to(device)
        out = net(img) #[batch,time]



