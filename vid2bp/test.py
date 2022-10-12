import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import wandb
import vid2bp.preprocessing.utils.signal_utils as su
import vid2bp.postprocessing.post_signal_utils as psu
import numpy as np


def test(model_n, model, test_loader, loss, idxx):
    model.eval()
    plot_flag = True
    test_cost_sum = 0
    n_cost_sum = 0
    d_cost_sum = 0
    s_cost_sum = 0
    if model_n == 'Unet':
        loss_m = loss
        with tqdm(test_loader, desc='Test', total=len(test_loader), leave=True) as test_epoch:
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
        loss_n = loss[0]
        loss_d = loss[1]
        loss_s = loss[2]

        with tqdm(test_loader, desc='Test', total=len(test_loader), leave=True) as test_epoch:
            idx = 0
            with torch.no_grad():
                for X_test, Y_test, d, s in test_epoch:
                    idx += 1
                    hypothesis = torch.squeeze(model(X_test[0]))
                    pred_d = torch.squeeze(model(X_test[1]))
                    pred_s = torch.squeeze(model(X_test[2]))

                    neg_cost = loss_n(hypothesis, Y_test)
                    d_cost = loss_d(pred_d, d)
                    s_cost = loss_s(pred_s, s)
                    cost = neg_cost + d_cost + s_cost
                    test_epoch.set_postfix(loss=cost.item())

                    test_cost_sum += cost
                    test_avg_cost = test_cost_sum / idx
                    n_cost_sum += neg_cost
                    n_avg_cost = n_cost_sum / idx
                    d_cost_sum += d_cost
                    d_avg_cost = d_cost_sum / idx
                    s_cost_sum += s_cost
                    s_avg_cost = s_cost_sum / idx
                    wandb.log({"Test Loss": test_avg_cost,
                               "Test negative Loss": n_avg_cost,
                               "Test Systolic Loss": s_avg_cost,
                               "Test Diastolic Loss": d_avg_cost}, step=idxx)

                    if plot_flag:
                        plot_flag = False
                        N = 0
                        h = hypothesis[N].cpu().detach()
                        y = Y_test[N].cpu().detach()
                        # plt.title("Epoch :" + str(idx + 1))
                        plt.subplot(3, 1, 1)
                        plt.plot(y)
                        plt.title("Epoch :" + str(idxx + 1) + "\nTarget ( s:" +
                                  str(np.round(np.mean(su.get_systolic(y.detach().cpu().numpy())), 2)) + " / d:" +
                                  str(np.round(np.mean(su.get_diastolic(y.detach().cpu().numpy())), 2)) + ")")
                        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.70)

                        plt.subplot(3, 1, 2)
                        plt.plot(psu.shape_scaler(h))
                        plt.title("Correlation")
                        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.70)
                        plt.subplot(3, 1, 3)
                        plt.plot(y)
                        plt.plot(psu.feature_combiner(h, pred_s, pred_d))
                        plt.title(
                            "Prediction ( s:" + str(np.round(pred_s.detach().cpu()[N].__float__(), 2)) + " / d:"
                            + str(np.round(pred_d.detach().cpu()[N].__float__(), 2)) + ")")
                        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.70)
                        wandb.log({"Prediction": wandb.Image(plt)})
                        plt.show()

