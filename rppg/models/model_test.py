import torch
from rppg.models.PhysNet import PhysNet as PhysNet

model_name = "PhysNet"

if __name__ == '__main__':

    if model_name == "PhysNet":
        img = torch.rand(32,3,32,32,32) # [batch, channel, length, width, height]
        net = PhysNet()
        out = net(img) #[batch, length]

