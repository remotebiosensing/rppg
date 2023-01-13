import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import wandb
# import vid2bp.preprocessing.utils.signal_utils as su
import vid2bp.postprocessing.post_signal_utils as psu
import numpy as np
from matplotlib import gridspec
from preprocessing.utils.signal_utils import ds_detection
import vid2bp.utils.train_utils as tu
from sklearn.preprocessing import MinMaxScaler
from vid2bp.nets.loss.loss import SelfScaler


def test(model, dataset, loss_list, epoch, scaler=True, plot_target=False):
    model.eval()
    scale_loss = SelfScaler().to('cuda:0')

    plot_flag = True
    avg_cost_list = []
    dy_avg_cost_list = []
    ddy_avg_cost_list = []
    for _ in range(len(loss_list)):
        avg_cost_list.append(0)
        dy_avg_cost_list.append(0)
        ddy_avg_cost_list.append(0)

    with tqdm(dataset, desc='Test{}'.format(str(epoch)), total=len(dataset), leave=True) as test_epoch:
        with torch.no_grad():
            for idx, (X_test, dx, ddx, Y_test, dy, ddy, d, s) in enumerate(test_epoch):
                hypothesis, dy_hypothesis, ddy_hypothesis, scaled_ple = model(X_test, dx, ddx, scaler=scaler)
                avg_cost_list, _ = tu.calc_losses(avg_cost_list, loss_list,
                                                  hypothesis, Y_test, idx + 1)
                dy_avg_cost_list, _ = tu.calc_losses(dy_avg_cost_list, loss_list,
                                                     dy_hypothesis, dy, idx + 1)
                ddy_avg_cost_list, _ = tu.calc_losses(ddy_avg_cost_list, loss_list,
                                                      ddy_hypothesis, ddy, idx + 1)

                ple_cost = scale_loss(scaled_ple, X_test)
                total_cost = torch.sum(torch.tensor(avg_cost_list)) + torch.sum(
                    torch.tensor(dy_avg_cost_list)) + torch.sum(torch.tensor(ddy_avg_cost_list)) + \
                             ple_cost

                postfix_dict = {}
                for i in range(len(loss_list)):
                    postfix_dict[(str(loss_list[i]))[:-2]] = (round(avg_cost_list[i], 3))
                postfix_dict['scale_variance'] = round(ple_cost.__float__(), 3)
                test_epoch.set_postfix(losses=postfix_dict, tot=total_cost)

                if plot_flag:
                    plot = plot_prediction(X_test[0], Y_test[0], [d[0], s[0]],
                                           hypothesis[0], epoch, plot_target)
                    plot_flag = False
            # wandb.init(project="VBPNet", entity="paperchae")
            # wandb.log({"Test Loss": total_cost}, step=epoch)
            # wandb.log({"Test Loss": test_avg_cost,
            #            'Pearson Loss': neg_cost,
            #            'STFT Loss': stft_cost}, step=epoch)
        return total_cost.__float__(), plot


def plot_prediction(ple, abp, dsm, hypothesis, epoch, plot_target):
    t = np.arange(0, 6, 1 / 60)
    h = np.squeeze(hypothesis.cpu().detach())
    y = abp.detach().cpu().numpy()
    x = ple.detach().cpu().numpy()
    min_max_scaler = MinMaxScaler()
    mean_sbp, mean_dbp, mean_map, sbp_idx, dbp_idx = ds_detection(h)
    target_sbp, target_dbp, target_map, target_sbp_idx, target_dbp_idx = ds_detection(y)
    # time = np.arange(0, 6, 0.01)
    '''fft version'''
    # fig = plt.figure(figsize=(15, 10))
    plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[2, 1])
    ax0 = plt.subplot(gs[0])
    if plot_target:
        ax0.plot(t, y, label='Target ABP')
        ax0.plot(t[target_sbp_idx], y[target_sbp_idx], 'r^', label='Target SBP')
        ax0.plot(t[target_dbp_idx], y[target_dbp_idx], 'bv', label='Target DBP')
        ax0.plot(t, h, label='Predicted ABP')
        ax0.plot(t[sbp_idx], h[sbp_idx], 'r^', label='Prediction SBP')
        ax0.plot(t[dbp_idx], h[dbp_idx], 'gv', label='Prediction DBP')
    else:
        ax0.plot(t, min_max_scaler.fit_transform(np.reshape(y, (360, 1))), label='Target ABP')
        # ax0.plot(t[target_sbp_idx], y[target_sbp_idx], 'r^', label='Target SBP')
        # ax0.plot(t[target_dbp_idx], y[target_dbp_idx], 'bv', label='Target DBP')
        ax0.plot(t, min_max_scaler.fit_transform(np.reshape(h, (360, 1))), label='Predicted ABP')
        # ax0.plot(t[sbp_idx], h[sbp_idx], 'r^', label='Prediction SBP')
        # ax0.plot(t[dbp_idx], h[dbp_idx], 'gv', label='Prediction DBP')
    ax0.set_title("Epoch :" + str(epoch + 1) +
                  "\nTarget ( s:" + str(np.round(dsm[1].detach().cpu().numpy(), 2)) +
                  " / d:" + str(np.round(dsm[0].detach().cpu().numpy(), 2)) + ")" +
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
    # wandb.log({"Prediction": wandb.Image(plt)})
    # plot = wandb.Image(plt)
    # plt.show()
    # plt.close()
    return plt
