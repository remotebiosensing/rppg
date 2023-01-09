from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from vid2bp.nets.modules.MultiResUNet1D import MultiResUNet1D
from vid2bp.nets.modules.UNetDS64 import UNetDS64
from vid2bp.PPG2ABP.dataset.PPG2ABP_dataset_loader import dataset_loader
from loss import NegPearsonLoss


def main():
    batch_size = 1
    length = 352

    # load weights
    app_model = UNetDS64(length=length).cuda()
    app_model.load_state_dict(torch.load('/home/najy/PycharmProjects/PPG2ABP_weights/UNetDS64_3.pth'))
    ref_model = MultiResUNet1D().cuda()
    ref_model.load_state_dict(torch.load('/home/najy/PycharmProjects/PPG2ABP_weights/MultiResUNet1D_3.pth'))

    # load data
    data_loaders = dataset_loader(channel=1, batch_size=batch_size)
    criterion1 = torch.nn.L1Loss()
    criterion2 = torch.nn.MSELoss()
    criterion3 = NegPearsonLoss()

    l1_loss_app = 0
    mse_loss_app = 0
    pearson_loss_app = 0

    l1_loss_ref = 0
    mse_loss_ref = 0
    pearson_loss_ref = 0

    max_abp, min_abp, max_ppg, min_ppg = data_loaders[0].dataset.get_max_min()

    for ppg, abp_out, abp_level1, abp_level2, abp_level3, abp_level4 in tqdm(data_loaders[2]):
        abp_signal_approximate = app_model(ppg) * (max_abp.to('cuda') - min_abp.to('cuda')) + min_abp.to('cuda')
        abp_signal_refined = ref_model(ppg) * (max_abp.to('cuda') - min_abp.to('cuda')) + min_abp.to('cuda')
        abp_signal = abp_out * (max_abp.to('cuda') - min_abp.to('cuda')) + min_abp.to('cuda')

        l1_loss_app += criterion1(abp_signal_approximate, abp_signal)
        mse_loss_app += criterion2(abp_signal_approximate, abp_signal)
        pearson_loss_app += criterion3(abp_signal_approximate, abp_signal)

        l1_loss_ref += criterion1(abp_signal_refined, abp_signal)
        mse_loss_ref += criterion2(abp_signal_refined, abp_signal)
        pearson_loss_ref += criterion3(abp_signal_refined, abp_signal)

        ppg_signal = ppg.detach().cpu().numpy()[0].squeeze() * (max_ppg - min_ppg) + min_ppg
        abp_signal = abp_out.detach().cpu().numpy()[0].squeeze() * (max_abp - min_abp) + min_abp
        abp_signal_approximate = abp_signal_approximate.detach().cpu().numpy()[0].squeeze()
        abp_signal_refined = abp_signal_refined.detach().cpu().numpy()[0].squeeze()

        plt.figure(figsize=(30, 15))
        plt.subplot(5, 1, 1)
        plt.plot(ppg_signal, label='PPG')
        plt.title("PPG")

        plt.subplot(5, 1, 2)
        plt.plot(abp_signal, label='ABP')
        plt.title("ABP")

        plt.subplot(5, 1, 3)
        plt.plot(abp_signal_approximate, label='ABP Approximate')
        plt.title("ABP Approximate")

        plt.subplot(5, 1, 4)
        plt.plot(abp_signal_refined, label='ABP Refined')
        plt.title("ABP Refined")

        plt.subplot(5, 1, 5)
        plt.plot(abp_signal, label='ABP', color='r')
        plt.plot(abp_signal_approximate, label='ABP Approximate', color='g')
        plt.plot(abp_signal_refined, label='ABP Refined', color='b')
        plt.title("ABP, ABP Approximate, ABP Refined")
        plt.legend()
        plt.tight_layout()
        plt.show()

    print('L1 Loss App: ', l1_loss_app / len(data_loaders[2]))
    print('MSE Loss App: ', mse_loss_app / len(data_loaders[2]))
    print('Pearson Loss App: ', pearson_loss_app / len(data_loaders[2]))

    print('L1 Loss Ref: ', l1_loss_ref / len(data_loaders[2]))
    print('MSE Loss Ref: ', mse_loss_ref / len(data_loaders[2]))
    print('Pearson Loss Ref: ', pearson_loss_ref / len(data_loaders[2]))