from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from vid2bp.nets.modules.LSTMAutoEncoder import LSTMAutoEncoder
from vid2bp.LSTMAutoEncoder.dataset.LSTMAutoEncoder_dataset_loader import dataset_loader
from loss import NegPearsonLoss


def main(plot_flag=False,
         weight_path='/home/najy/PycharmProjects/PPG2ABP_weights/MultiResUNet1D_0.017421271360944957.pth',
         dataset_root_path="/home/najy/PycharmProjects/PPG2ABP_datasets/preprocessed/"):
    batch_size = 2

    # load weights
    model = LSTMAutoEncoder().cuda()
    model.load_state_dict(torch.load(weight_path))

    # load data
    data_loaders, meta_params = dataset_loader(batch_size=batch_size, label='abp', dataset_root_path=dataset_root_path)
    criterion1 = torch.nn.L1Loss()
    criterion2 = torch.nn.MSELoss()
    criterion3 = NegPearsonLoss()

    l1_loss_app = 0
    mse_loss_app = 0
    pearson_loss_app = 0

    l1_loss_ref = 0
    mse_loss_ref = 0
    pearson_loss_ref = 0

    train_ple_std, train_ple_mean, train_abp_std, train_abp_mean = meta_params

    for ppg, abp in tqdm(data_loaders[2]):
        app_pred = model(ppg)
        abp_signal_pred = app_pred * train_abp_std + train_abp_mean
        abp_signal = abp * train_abp_std + train_abp_mean

        l1_loss_app += criterion1(abp_signal_approximate, abp_signal).__float__()
        mse_loss_app += criterion2(abp_signal_approximate, abp_signal).__float__()
        pearson_loss_app += criterion3(abp_signal_approximate, abp_signal).__float__()

        l1_loss_ref += criterion1(abp_signal_refined, abp_signal).__float__()
        mse_loss_ref += criterion2(abp_signal_refined, abp_signal).__float__()
        pearson_loss_ref += criterion3(abp_signal_refined, abp_signal).__float__()

        if plot_flag:
            ppg_signal = ppg.detach().cpu().numpy()[0].squeeze() * (max_ppg - min_ppg) + min_ppg
            abp_signal = abp_out.detach().cpu().numpy()[0].squeeze() * (max_abp - min_abp) + min_abp
            abp_signal_approximate = abp_signal_approximate.detach().cpu().numpy()[0].squeeze()
            abp_signal_refined = abp_signal_refined.detach().cpu().numpy()[0].squeeze()

            plt.figure(figsize=(30, 15))
            plt.subplot(5, 1, 1)
            plt.plot(ppg_signal, label='PPG')
            plt.title("PPG")

            plt.subplot(5, 1, 2)
            plt.plot(abp_signal, label='ABP', c='r')
            plt.title("Target ABP")

            plt.subplot(5, 1, 3)
            plt.plot(abp_signal_approximate, label='ABP Approximate', c='g')
            plt.title("ABP Approximate")

            plt.subplot(5, 1, 4)
            plt.plot(abp_signal_refined, label='ABP Refined', c='b')
            plt.title("ABP Refinement")

            plt.subplot(5, 1, 5)
            plt.plot(abp_signal, label='ABP', color='r')
            plt.plot(abp_signal_approximate, label='ABP Approximate', color='g')
            plt.plot(abp_signal_refined, label='ABP Refined', color='b')
            plt.title("ABP, ABP Approximate, ABP Refinement")
            plt.legend()
            plt.tight_layout()
            plt.show()

    print('L1 Loss App: ', l1_loss_app / len(data_loaders[2]))
    print('MSE Loss App: ', mse_loss_app / len(data_loaders[2]))
    print('Pearson Loss App: ', pearson_loss_app / len(data_loaders[2]))

    print('L1 Loss Ref: ', l1_loss_ref / len(data_loaders[2]))
    print('MSE Loss Ref: ', mse_loss_ref / len(data_loaders[2]))
    print('Pearson Loss Ref: ', pearson_loss_ref / len(data_loaders[2]))


if __name__ == '__main__':
    main(True)
