import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
from matplotlib import gridspec
# from preprocessing.utils.signal_utils import ds_detection
from sklearn.preprocessing import MinMaxScaler


def test(model, dataset, loss_list, epoch, scaler=True, plot_scaled=False, patient_information=None):
    model.eval()
    # scale_loss = SelfScaler().to('cuda:0')
    # mape_loss = MAPELoss().to('cuda:0')
    # neg_loss = NegPearsonLoss().to('cuda:0')
    #
    plot_flag = True
    cost_sum = 0
    dbp_cost_sum = 0
    sbp_cost_sum = 0
    scale_cost_sum = 0
    total_cost_sum = 0

    # avg_cost_list = []
    # dy_avg_cost_list = []
    # ddy_avg_cost_list = []
    # for _ in range(len(loss_list)):
    #     avg_cost_list.append(0)
    #     dy_avg_cost_list.append(0)
    #     ddy_avg_cost_list.append(0)

    with tqdm(dataset, desc='Test-{}'.format(str(epoch)), total=len(dataset), leave=True) as test_epoch:
        with torch.no_grad():
            for idx, (X_test, Y_test, d, s, m, info, ohe) in enumerate(test_epoch):
                hypothesis, dbp, sbp, amp = model(X_test)
                cost = loss_list[0](hypothesis, Y_test)
                dbp_cost, sbp_cost, scale_cost = loss_list[-1](dbp, sbp, amp, d, s, amp)

                # amp_cost = loss_list[-1](dbp, sbp, mbp, d, s, m)
                total_cost = cost + dbp_cost + sbp_cost + scale_cost
                # total_cost = cost + dbp_cost + sbp_cost + scale_cost# + amp_cost_sum

                cost_sum += cost.item()
                avg_cost = cost_sum / (idx + 1)
                dbp_cost_sum += dbp_cost.item()
                dbp_avg_cost = dbp_cost_sum / (idx + 1)
                sbp_cost_sum += sbp_cost.item()
                sbp_avg_cost = sbp_cost_sum / (idx + 1)
                scale_cost_sum += scale_cost.item()
                scale_avg_cost = scale_cost_sum / (idx + 1)
                # amp_cost_sum += amp_cost.item()
                # amp_avg_cost = amp_cost_sum / (idx + 1)
                total_cost_sum += total_cost.item()
                total_avg_cost = total_cost_sum / (idx + 1)

                postfix_dict = {}

                postfix_dict['corr'] = round(avg_cost, 3)
                # postfix_dict['amp'] = round(amp_avg_cost, 3)
                postfix_dict['dbp'] = round(dbp_avg_cost, 3)
                postfix_dict['sbp'] = round(sbp_avg_cost, 3)
                postfix_dict['scale'] = round(scale_avg_cost, 3)
                postfix_dict['total'] = round(total_avg_cost, 3)

                test_epoch.set_postfix(losses=postfix_dict)

                if plot_flag:
                    sub = torch.abs(dbp - d)
                    x = np.array(range(30, 220))
                    plt.figure(figsize=(10, 10))
                    plt.xlim(30, 220)
                    plt.ylim(30, 220)
                    plt.title('Predicted ABP Scatter Plot')
                    plt.scatter(torch.squeeze(d).detach().cpu(), torch.squeeze(dbp).detach().cpu(), c='blue',
                                marker='x', alpha=0.2, label='DBP')
                    plt.scatter(torch.squeeze(s).detach().cpu(), torch.squeeze(sbp).detach().cpu(), c='red',
                                alpha=0.2, label='SBP')
                    plt.plot(x, x, color='k', label='y=x')
                    plt.grid(color='gray', alpha=.5, linestyle='--')
                    plt.xlabel('Target BP')
                    plt.ylabel('Predicted BP')
                    plt.legend()
                    plt.show()
                    # plt.close()
                    scatter_plot = plt
                    # plt.close()
                    plot = plot_prediction(X_test[0][0], Y_test[0], [d[0], s[0]],
                                           hypothesis[0], dbp[0], sbp[0], epoch, plot_scaled, info[0],
                                           patient_information, 1 - postfix_dict['corr'])
                    plot_flag = False

        return total_avg_cost, plot, scatter_plot


def plot_prediction(ple, abp, dsm, hypothesis, d, s, epoch, scaled, p_info, patient_information, corr):
    t = np.arange(0, 6, 1 / 60)
    h = np.squeeze(hypothesis.cpu().detach())
    x = ple.detach().cpu().numpy()
    y = abp.detach().cpu().numpy()
    info = p_info.detach().cpu().numpy()
    min_max_scaler = MinMaxScaler()
    info_list = np.squeeze(patient_information[patient_information['HADM_ID'] == int(info[1])].to_numpy())
    subject_id, gender, ethnicity, diag, diag_group = info_list[1], info_list[3], info_list[4], info_list[5], info_list[
        6]
    # mean_sbp, mean_dbp, mean_map, sbp_idx, dbp_idx = ds_detection(h)
    # target_sbp, target_dbp, target_map, target_sbp_idx, target_dbp_idx = ds_detection(y)
    # h_abp_info = su.BPInfoExtractor(h)
    # h_sbp, h_dbp = h_abp_info.sbp, h_abp_info.dbp
    # y_abp_info = su.BPInfoExtractor(y)
    # y_sbp, y_dbp = y_abp_info.sbp, y_abp_info.dbp
    # time = np.arange(0, 6, 0.01)
    '''fft version'''
    # fig = plt.figure(figsize=(15, 10))
    plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[2, 1])
    ax0 = plt.subplot(gs[0])
    if scaled:
        ax0.plot(t, y, label='Target ABP')
        # ax0.plot(t[y_sbp[-1][-1]], y[y_sbp[-1][-1]], 'r^', label='Target SBP')
        # ax0.plot(t[y_dbp[-1]], y[y_dbp[-1]], 'bv', label='Target DBP')
        ax0.plot(t, h, label='Predicted ABP')
        # ax0.plot(t[h_sbp[-1][-1]], h[h_sbp[-1][-1]], 'r^', label='Prediction SBP')
        # ax0.plot(t[h_dbp[-1]], h[h_dbp[-1]], 'gv', label='Prediction DBP')
    else:
        ax0.plot(t, y, label='Target ABP')
        # ax0.plot(t[y_sbp[-1][-1]], y[y_sbp[-1][-1]], 'r^', label='Target SBP')
        # ax0.plot(t[y_dbp[-1]], y[y_dbp[-1]], 'bv', label='Target DBP')
        # ax0.plot(t, min_max_scaler.fit_transform(np.reshape(y, (360, 1))), label='Target ABP')
        ax0.plot(t, h, label='Predicted ABP')
        # ax0.plot(t[y_sbp[-1][-1]], h[y_sbp[-1][-1]], 'r^', label='Prediction SBP')
        # ax0.plot(t[y_dbp[-1]], h[y_dbp[-1]], 'gv', label='Prediction DBP')
        # ax0.plot(t, min_max_scaler.fit_transform(np.reshape(h, (360, 1))), label='Predicted ABP')
    title = "Patient Info : [ID : P00{} / Gender : {} / Ethnicity : {}] Corr : {}".format(str(subject_id),
                                                                                          str(gender),
                                                                                          str(ethnicity),
                                                                                          str(np.round(corr, 2))) + \
            "\nDiagnosis Group(Diagnosis) : {}({})".format(str(diag_group), str(diag)) + \
            "\nTarget (SBP: {}mmHg / DBP : {}mmHg) ".format(str(np.round(dsm[1].detach().cpu().numpy(), 2)),
                                                            str(np.round(dsm[0].detach().cpu().numpy(), 2))) + \
            "Prediction (SBP: {}mmHg / DBP : {}mmHg)".format(str(np.round(s.detach().cpu().numpy(), 2)),
                                                             str(np.round(d.detach().cpu().numpy(), 2)))
    # ax0.set_title(
    #     "Patient Info : " + '[ID :p00' + str(subject_id) + ' / Gender : ' + str(gender) + ' / Ethnicity : ' + str(
    #         ethnicity) + ' / corr : ' + str(1 - corr) +
    #     ']\nDiagnosis Group(Diagnosis) :' + str(diag_group) + '(' + str(diag) + ')' +
    #     "\nTarget ( Systolic:" + str(np.round(dsm[1].detach().cpu().numpy(), 2)) +
    #     "mmHg / Diastolic:" + str(np.round(dsm[0].detach().cpu().numpy(), 2)) + "mmHg)" +
    #     "  Prediction ( Systolic:" + str(np.round(s.detach().cpu().numpy()[0], 2)) +
    #     "mmHg / Diastolic:" + str(np.round(d.detach().cpu().numpy()[0], 2)) + "mmHg)")
    ax0.set_title(title)

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
