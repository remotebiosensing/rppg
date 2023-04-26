import torch
from rppg.models.PhysNet import PhysNet as PhysNet
from rppg.models.DeepPhys import DeepPhys as DeepPhys
from rppg.models.ETArPPGNet import ETArPPGNet as ETArPPGNet

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
        #TODO : need to check input shape
        img = torch.rand(32,5,3, 60, 50, 50).to(device) #[batch, block, Channel, time, width, height]
        net = ETArPPGNet().to(device)
        out = net(img)


