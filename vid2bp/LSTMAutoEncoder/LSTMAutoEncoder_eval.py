from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from vid2bp.nets.modules.LSTMAutoEncoder import LSTMAutoEncoder
from vid2bp.LSTMAutoEncoder.dataset.LSTMAutoEncoder_dataset_loader import dataset_loader
from loss import NegPearsonLoss


def main(plot_flag=False,
         weight_path='/home/najy/rppg/model_weights/LSTMAutoEncoder_predict_abp_49.pth',
         dataset_root_path="/home/najy/PycharmProjects/vid2bp_datasets/raw/",
         batch_size=2):
    # load weights
    model = LSTMAutoEncoder(hidden_size=128, input_size=3, output_size=1, label='abp').cuda()
    model.load_state_dict(torch.load(weight_path))

    # load data
    data_loaders, meta_params = dataset_loader(batch_size=batch_size, label='abp', dataset_root_path=dataset_root_path)
    train_ple_std, train_ple_mean = meta_params

    criterion1 = torch.nn.L1Loss()
    criterion2 = torch.nn.MSELoss()
    criterion3 = NegPearsonLoss()

    l1_loss = 0
    mse_loss = 0
    pearson_loss = 0

    train_abp_std, train_abp_mean = data_loaders[0].dataset.get_mean_std()
    test_abp_std, test_abp_mean = data_loaders[2].dataset.get_mean_std()

    for ppg, abp in tqdm(data_loaders[2]):
        app_pred = model(ppg)
        abp_signal_pred = app_pred * train_abp_std + train_abp_mean
        abp_signal_target = abp * test_abp_std + test_abp_mean

        l1_loss += criterion1(abp_signal_pred, abp_signal_target).__float__()
        mse_loss += criterion2(abp_signal_pred, abp_signal_target).__float__()
        pearson_loss += criterion3(abp_signal_pred, abp_signal_target).__float__()

        if plot_flag:
            ppg_signal = ppg.detach().cpu().numpy()[0].squeeze() * train_ple_std + train_ple_mean
            abp_signal_target = abp_signal_target.detach().cpu().numpy()[0].squeeze()
            abp_signal_pred = abp_signal_pred.detach().cpu().numpy()[0].squeeze()

            plt.clf()
            plt.figure(figsize=(30, 15))
            plt.subplot(4, 1, 1)
            plt.plot(ppg_signal[:, 0], label='PPG')
            plt.title("PPG")

            plt.subplot(4, 1, 2)
            plt.plot(abp_signal_target, label='ABP', c='r')
            plt.title("Target ABP")

            plt.subplot(4, 1, 3)
            plt.plot(abp_signal_pred, label='ABP Approximate', c='g')
            plt.title("Prediction ABP")

            plt.subplot(4, 1, 4)
            plt.plot(abp_signal_target, label='ABP', color='r')
            plt.plot(abp_signal_pred, label='ABP Approximate', color='g')
            plt.title("ABP Target, ABP Prediction")
            plt.legend()
            plt.tight_layout()
            plt.show()

    print('L1 Loss: ', l1_loss / len(data_loaders[2]))
    print('MSE Loss: ', mse_loss / len(data_loaders[2]))
    print('Pearson Loss: ', pearson_loss / len(data_loaders[2]))


if __name__ == '__main__':
    main(plot_flag=False,
         weight_path='/home/najy/rppg/model_weights/LSTMAutoEncoder_predict_abp_49.pth',
         dataset_root_path="/home/najy/PycharmProjects/vid2bp_datasets/raw/",
         batch_size=128)
