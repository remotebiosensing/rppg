import torch
import matplotlib.pyplot as plt
from vid2bp.nets.modules.MultiResUNet1D import MultiResUNet1D
from vid2bp.PPG2ABP.dataset.PPG2ABP_dataset_loader import dataset_loader

learning_rate = 0.001
batch_size = 1
epochs = 100
length = 352

# load weights
# model = UNetDS64.UNetDS64(length=length).cuda()
model = MultiResUNet1D().cuda()
model.load_state_dict(torch.load('/home/najy/PycharmProjects/PPG2ABP_weights/MultiResUNet1D_3.pth'))

# load data
data_loaders = dataset_loader(channel=1, batch_size=batch_size)
criterion = torch.nn.MSELoss()
for ppg, abp_out, abp_level1, abp_level2, abp_level3, abp_level4 in data_loaders[2]:
    ppg = ppg.cuda()
    abp_out = abp_out.cuda()
    # abp_level1 = abp_level1.cuda()
    # abp_level2 = abp_level2.cuda()
    # abp_level3 = abp_level3.cuda()
    # abp_level4 = abp_level4.cuda()

    pred_out = model(ppg)

    loss = criterion(pred_out, abp_out)

    pred_out = pred_out.detach().cpu().numpy()[0].squeeze()
    abp_out = abp_out.detach().cpu().numpy()[0].squeeze()
    plt.title("test loss: " + str(loss.item()))
    plt.plot(pred_out, label='pred')
    plt.plot(abp_out, label='gt')
    plt.legend()
    plt.show()


