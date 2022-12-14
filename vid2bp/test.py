import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import wandb
import vid2bp.preprocessing.utils.signal_utils as su
import vid2bp.postprocessing.post_signal_utils as psu
import numpy as np
from matplotlib import gridspec


def test(model_n, model, test_loader, loss, idxx):
    model.eval()
    plot_flag = True
    test_cost_sum = 0
    # n_cost_sum = 0
    # d_cost_sum = 0
    # s_cost_sum = 0
    if model_n == 'Unet':
        loss_m = loss
        with tqdm(test_loader, desc='Test{}'.format(str(idxx)), total=len(test_loader), leave=True) as test_epoch:
            idx = 0
            with torch.no_grad():
                for X_test, Y_test in test_epoch:
                    idx += 1
                    hypothesis = torch.squeeze(model(X_test))
                    cost = loss_m(hypothesis, Y_test)
                    test_epoch.set_postfix(loss=cost.item())
                    test_cost_sum += cost
                    test_avg_cost = test_cost_sum / idx
                    wandb.log({"Test MSE loss": test_avg_cost}, step=idxx)
                    if plot_flag:
                        plot_flag = False
                        N = 0
                        h = hypothesis[N].cpu().detach()
                        y = Y_test[N].cpu().detach()
                        plt.subplot(2, 1, 1)
                        plt.plot(y)
                        plt.title("Epoch :" + str(idxx + 1) + "\nTarget ( s:" +
                                  str(np.round(np.mean(su.get_systolic(y.detach().cpu().numpy())), 2)) + " / d:" +
                                  str(np.round(np.mean(su.get_diastolic(y.detach().cpu().numpy())), 2)) + ")")
                        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.70)

                        plt.subplot(2, 1, 2)
                        plt.plot(y)
                        plt.plot(h)
                        plt.title("Prediction")
                        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.70)
                        wandb.log({"Prediction": wandb.Image(plt)})
                        plt.show()

    elif model_n == 'BPNet':
        # loss_n = loss[0]
        # loss_d = loss[1]
        # loss_s = loss[2]
        loss_fft = loss[0]

        with tqdm(test_loader, desc='Test{}'.format(str(idxx)), total=len(test_loader), leave=True) as test_epoch:
            idx = 0
            with torch.no_grad():
                for X_test, Y_test, d, s in test_epoch:
                    idx += 1
                    hypothesis = torch.squeeze(model(X_test))
                    # pred_d = torch.squeeze(model(X_test)[1])
                    # pred_s = torch.squeeze(model(X_test)[2])

                    # neg_cost = loss_n(hypothesis, Y_test)
                    # d_cost = loss_d(pred_d, d)
                    # s_cost = loss_s(pred_s, s)
                    fft_cost = loss_fft(hypothesis, Y_test)
                    # cost = neg_cost + d_cost + s_cost
                    cost = fft_cost
                    test_epoch.set_postfix(loss=cost.item())

                    test_cost_sum += cost
                    test_avg_cost = test_cost_sum / idx
                    # n_cost_sum += neg_cost
                    # n_avg_cost = n_cost_sum / idx
                    # d_cost_sum += d_cost
                    # d_avg_cost = d_cost_sum / idx
                    # s_cost_sum += s_cost
                    # s_avg_cost = s_cost_sum / idx
                    # wandb.log({"Test Loss": test_avg_cost,
                    #            "Test Negative Pearson Loss": n_avg_cost,
                    #            "Test Systolic Loss": s_avg_cost,
                    #            "Test Diastolic Loss": d_avg_cost}, step=idxx)
                    wandb.log({"Test FFT Loss": test_avg_cost}, step=idxx)

                    if plot_flag:
                        plot_flag = False
                        t = np.arange(0, 6, 1 / 60)
                        h = hypothesis[0].cpu().detach()
                        y = Y_test[0].cpu().detach()
                        x = X_test[0][0].cpu().detach()
                        # time = np.arange(0, 6, 0.01)
                        '''fft version'''
                        # fig = plt.figure(figsize=(15, 10))
                        plt.figure(figsize=(15, 10))
                        gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[2, 1])
                        ax0 = plt.subplot(gs[0])
                        ax0.plot(t, y, label='Target ABP')
                        ax0.plot(t, h, label='Prediction ABP')
                        ax0.set_title("Epoch :" + str(idxx + 1) +
                                      "\nTarget ( s:" +
                                      str(np.round(np.mean(su.get_systolic(y.detach().cpu().numpy())[-1]),
                                                   2)) + " / d:" +
                                      str(np.round(np.mean(su.get_diastolic(y.detach().cpu().numpy())), 2)) + ")" +
                                      "  Prediction ( s:" +
                                      str(np.round(np.mean(su.get_systolic(h.detach().cpu().numpy())[-1]),
                                                   2)) + " / d:" +
                                      str(np.round(np.mean(su.get_diastolic(h.detach().cpu().numpy())), 2)) + ")")
                        # ax0.set_xlabel('Time (s)')
                        ax0.set_ylabel('Arterial Blood Pressure (mmHg)')
                        ax0.legend(loc='upper right')
                        ax1 = plt.subplot(gs[1])
                        ax1.plot(t, x, linestyle='dashed', label='Input PPG')
                        ax1.set_xlabel('Time (Sec, 60fps)')
                        ax1.set_ylabel('Photoplethysmography')
                        ax1.legend(loc='upper right')

                        # plt.title("Epoch :" + str(idxx + 1) +
                        #           "\nTarget ( s:" +
                        #           str(np.round(np.mean(su.get_systolic(y.detach().cpu().numpy())[-1]), 2)) + " / d:" +
                        #           str(np.round(np.mean(su.get_diastolic(y.detach().cpu().numpy())), 2)) + ")" +
                        #           "\tPrediction ( s:" +
                        #           str(np.round(np.mean(su.get_systolic(h.detach().cpu().numpy())[-1]), 2)) + " / d:" +
                        #           str(np.round(np.mean(su.get_diastolic(h.detach().cpu().numpy())), 2)) + ")")
                        # plt.plot(y, label='Target ABP')
                        # plt.plot(h, label='Predicted ABP')
                        # plt.xlabel('Time (Sec, 60fps)')
                        # plt.ylabel('Arterial Blood Pressure (mmHg)')
                        # plt.legend(loc='upper right')

                        ''' 1개로 나오는 version
                        plt.title("Epoch :" + str(idxx + 1) +
                                  "\nTarget ( s:" +
                                  str(np.round(np.mean(su.get_systolic(y.detach().cpu().numpy())[-1]), 2)) + " / d:" +
                                  str(np.round(np.mean(su.get_diastolic(y.detach().cpu().numpy())), 2)) + ")" +
                                  "\nPrediction ( s:" +
                                  str(np.round(pred_s.detach().cpu()[0].__float__(), 2)) + " / d:" +
                                  str(np.round(pred_d.detach().cpu()[0].__float__(), 2)) + ")")
                        plt.plot(y, label='Target')
                        plt.plot(psu.feature_combiner(h, pred_s, pred_d), label='Prediction')
                        plt.legend(loc='upper right')
                        '''
                        ''' 3개로 나오는 version
                        # plt.title("Epoch :" + str(idx + 1))
                        # plt.subplot(3, 1, 1)
                        # plt.plot(y)
                        # plt.title("Epoch :" + str(idxx + 1) + "\nTarget ( s:" +
                        #           str(np.round(np.mean(su.get_systolic(y.detach().cpu().numpy())), 2)) + " / d:" +
                        #           str(np.round(np.mean(su.get_diastolic(y.detach().cpu().numpy())), 2)) + ")")
                        # plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.70)
                        #
                        # plt.subplot(3, 1, 2)
                        # plt.plot(psu.shape_scaler(h))
                        # plt.title("Correlation")
                        # plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.70)
                        # plt.subplot(3, 1, 3)
                        # plt.plot(y)
                        # plt.plot(psu.feature_combiner(h, pred_s, pred_d))
                        # plt.title(
                        #     "Prediction ( s:" + str(np.round(pred_s.detach().cpu()[N].__float__(), 2)) + " / d:"
                        #     + str(np.round(pred_d.detach().cpu()[N].__float__(), 2)) + ")")
                        # plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.70)
                        '''
                        wandb.log({"Prediction": wandb.Image(plt)})
                        plt.close()
                        # plt.show()
