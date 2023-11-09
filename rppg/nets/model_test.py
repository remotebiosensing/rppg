import torch

from nets.APNETv2 import APNETv2 as APNETv2
from nets.DeepPhys import DeepPhys as DeepPhys
from nets.ETArPPGNet import ETArPPGNet as ETArPPGNet
from nets.PhysNet import PhysNet as PhysNet
from nets.EfficientPhys import EfficientPhys as EfficientPhys

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

# model_name = "APNETv2"
model_name = "EfficientPhys"


def check_APNETv2_computation_time(net):
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(11)]  # number of forward events
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(11)]
    img = torch.rand(3, 32, 64, 3, 30, 30).to(
        device)  # [fore head + left cheek + right cheek, batch, time, channel, width, height]
    # 0
    start_events[0].record()
    f_feature_map = net.f_m.main_seq_stem(img[0])
    end_events[0].record()
    # 1
    start_events[1].record()
    l_feature_map = net.l_m.main_seq_stem(img[1])
    end_events[1].record()
    # 2
    start_events[2].record()
    r_feature_map = net.r_m.main_seq_stem(img[2])
    end_events[2].record()
    # 3
    start_events[3].record()
    f_s = net.sub(torch.mean(img[0], dim=(3, 4)))
    end_events[3].record()
    # 4
    start_events[4].record()
    l_s = net.sub(torch.mean(img[1], dim=(3, 4)))
    end_events[4].record()
    # 5
    start_events[5].record()
    r_s = net.sub(torch.mean(img[2], dim=(3, 4)))
    end_events[5].record()

    combined_feature_maps = torch.cat([f_feature_map, l_feature_map, r_feature_map], dim=1)

    # 6
    start_events[6].record()
    fused_feature_maps = net.feature_fusion(combined_feature_maps)
    end_events[6].record()
    # 7
    start_events[7].record()
    f_out = net.f_m.forward_after_stem(fused_feature_maps)
    end_events[7].record()
    # 8
    start_events[8].record()
    l_out = net.l_m.forward_after_stem(fused_feature_maps)
    end_events[8].record()
    # 9
    start_events[9].record()
    r_out = net.r_m.forward_after_stem(fused_feature_maps)
    end_events[9].record()

    combined = torch.stack([f_out, l_out, r_out], dim=2)
    # 10
    start_events[10].record()
    attention_weights = net.attention(torch.mean(combined, dim=1))
    attention_weights = attention_weights.unsqueeze(1)
    end_events[10].record()

    torch.cuda.synchronize()

    for i in range(11):
        elapsed_time_ms = start_events[i].elapsed_time(end_events[i])
        print(i,f"elapsed time (ms): {elapsed_time_ms}")







if __name__ == '__main__':

    if model_name == "PhysNet":
        img = torch.rand(32, 3, 32, 32, 32).to(device)  # [batch, channel, length, width, height]
        net = PhysNet().to(device)
        out = net(img)  # [batch, length]
    elif model_name == "DeepPhys":
        img = torch.rand(32, 2, 3, 36, 36).to(device)  # [batch, norm + diff, channel, width, height]
        net = DeepPhys().to(device)
        out = net(img)  # [batch, 1]
    elif model_name == "ETArPPGNet":
        img = torch.rand(32, 30, 3, 10, 224, 224).to(device)  # [batch, block, Channel, time, width, height]
        net = ETArPPGNet().to(device)
        out = net(img)  # [batch, block * time]
    elif model_name == "APNETv2":
        img = torch.rand(3, 32, 64, 3, 30, 30).to(
            device)  # [fore head + left cheek + right cheek, batch, time, channel, width, height]
        net = APNETv2().to(device)
        # check_APNETv2_computation_time(net)
        out = net(img)  # [batch,time]
    elif model_name == "EfficientPhys":
        img = torch.rand(40, 3, 72, 72).to(device)  # [batch, channel, height, width]
        N, C, H, W = img.shape
        net = EfficientPhys(img_size=W).to(device)
        out = net(img)
