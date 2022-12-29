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
    h = np.squeeze(hypothesis[0].cpu().detach())
    y = abp[0].cpu().detach()
    x = ple[0].cpu().detach()
    mean_sbp, mean_dbp, mean_map, sbp_idx, dbp_idx = ds_detection(h)
    # target_sbp, target_dbp, target_map, target_sbp_idx, target_dbp_idx = ds_detection(y)
    # time = np.arange(0, 6, 0.01)
    '''fft version'''
    # fig = plt.figure(figsize=(15, 10))
    plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[2, 1])
    ax0 = plt.subplot(gs[0])
    # ax0.plot(t, y, label='Target ABP')
    # ax0.plot(y[target_sbp_idx], 'ro', label='Target SBP')
    # ax0.plot(y[target_dbp_idx], 'ro', label='Target DBP')
    ax0.plot(t, h, label='Predicted ABP')
    # ax0.plot(h[sbp_idx], 'go', label='Prediction SBP')
    # ax0.plot(h[dbp_idx], 'go', label='Prediction DBP')
    ax0.set_title("Epoch :" + str(epoch + 1) +
                  "\nTarget ( s:" + str(np.round(dsm[0][1].cpu().detach(), 2)) +
                  " / d:" + str(np.round(dsm[0][0].cpu().detach(), 2)) + ")" +
                  "  Prediction ( s:" + str(np.round(mean_sbp, 2)) +
                  " / d:" + str(np.round(mean_dbp, 2)) + ")")
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
            neg_cost = loss[0](hypothesis, Y_test)
            # neg_cost = 0
            '''STFT Loss'''
            stft_cost = loss[1](hypothesis, Y_test)
            '''DBP Loss'''
            # d_cost = loss[1](pred_dia, dia)
            '''SBP Loss'''
            # s_cost = loss[2](pred_sys, sys)

            '''Total Loss'''
            cost = neg_cost + stft_cost
            test_cost_sum += cost.__float__()
            test_avg_cost = test_cost_sum / (idx + 1)
            test_epoch.set_postfix(n=neg_cost.__float__(), s=stft_cost.__float__(), loss=test_avg_cost)
            if plot_flag:
                plot_prediction(X_test[0], Y_test, [dia, sys, mean], hypothesis, epoch)
                plot_flag = False
        wandb.log({"Test Loss": test_avg_cost}, step=epoch)
        # wandb.log({"Test Loss": test_avg_cost,
        #            'Pearson Loss': neg_cost,
        #            'STFT Loss': stft_cost}, step=epoch)
    return test_avg_cost.__float__()
