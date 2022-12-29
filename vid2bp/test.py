import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import wandb
# import vid2bp.preprocessing.utils.signal_utils as su
import vid2bp.postprocessing.post_signal_utils as psu
import numpy as np
from matplotlib import gridspec
from preprocessing.utils.signal_utils import ds_detection


def plot_prediction(ple, abp, dsm, hypothesis, epoch):

    t = np.arange(0, 6, 1 / 60)
    h = np.squeeze(hypothesis.cpu().detach())
    y = abp.cpu().detach()
    x = ple[0].cpu().detach()
    mean_sbp, mean_dbp, mean_map, sbp_idx, dbp_idx = ds_detection(h)
    target_sbp, target_dbp, target_map, target_sbp_idx, target_dbp_idx = ds_detection(y)
    # time = np.arange(0, 6, 0.01)
    '''fft version'''
    # fig = plt.figure(figsize=(15, 10))
    plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[2, 1])
    ax0 = plt.subplot(gs[0])
    ax0.plot(t, y, label='Target ABP')
    ax0.plot(t[target_sbp_idx], y[target_sbp_idx], 'r^', label='Target SBP')
    ax0.plot(t[target_dbp_idx], y[target_dbp_idx], 'bv', label='Target DBP')
    ax0.plot(t, h, label='Predicted ABP')
    ax0.plot(t[sbp_idx], h[sbp_idx], 'r^', label='Prediction SBP')
    ax0.plot(t[dbp_idx], h[dbp_idx], 'gv', label='Prediction DBP')
    ax0.set_title("Epoch :" + str(epoch + 1) +
                  "\nTarget ( s:" + str(np.round(dsm[1], 2).cpu().detach()) +
                  " / d:" + str(np.round(dsm[0].cpu().detach(), 2).cpu().detach()) + ")" +
                  "  Prediction ( s:" + str(np.round(mean_sbp, 2).detach().cpu()) +
                  " / d:" + str(np.round(mean_dbp, 2).detach().cpu()) + ")")
    # ax0.set_xlabel('Time (s)')
    ax0.set_ylabel('Arterial Blood Pressure (mmHg)')
    ax0.legend(loc='upper right')
    ax1 = plt.subplot(gs[1])
    ax1.plot(t, x, linestyle='dashed', label='Input PPG')
    ax1.set_xlabel('Time (Sec, 60fps)')
    ax1.set_ylabel('Photoplethysmography')
    ax1.legend(loc='upper right')
    wandb.log({"Prediction": wandb.Image(plt)})
    # plt.show()
    plt.close()


def test(model, dataset, loss, epoch, scaler=True):
    model.eval()
    plot_flag = True
    test_avg_cost = 0
    test_cost_sum = 0
    test_epoch = tqdm(dataset, desc='Test{}'.format(str(epoch)), total=len(dataset), leave=True)

    # with tqdm(dataset, desc='Test{}'.format(str(epoch)), total=len(dataset), leave=True) as test_epoch:
    with torch.no_grad():
        for idx, (X_test, Y_test, dia, sys, mean) in enumerate(test_epoch):
            hypothesis = model(X_test, scaler=scaler)

            '''Negative Pearson Loss'''
            rmse_cost = loss[0](hypothesis, Y_test)
            # neg_cost = 0
            '''STFT Loss'''
            stft_cost = loss[1](hypothesis, Y_test)
            '''DBP Loss'''
            # d_cost = loss[0](pred_d, dia)
            '''SBP Loss'''
            # s_cost = loss[0](pred_s, sys)

            '''Total Loss'''
            cost = rmse_cost + stft_cost #  + d_cost + s_cost
            test_cost_sum += cost.__float__()
            test_avg_cost = test_cost_sum / (idx + 1)
            test_epoch.set_postfix(rmse=rmse_cost.__float__(), stft=stft_cost.__float__(), tot=test_avg_cost)
            if plot_flag:
                plot_prediction(X_test[0], Y_test[0], [dia[0], sys[0], mean[0]], hypothesis[0], epoch)
                plot_flag = False
        wandb.log({"Test Loss": test_avg_cost}, step=epoch)
        # wandb.log({"Test Loss": test_avg_cost,
        #            'Pearson Loss': neg_cost,
        #            'STFT Loss': stft_cost}, step=epoch)
    return test_avg_cost.__float__()
