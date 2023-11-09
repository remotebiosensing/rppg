import matplotlib.pyplot as plt
import utils.funcs
import numpy as np


def hrv_comparison_plot(raw_tar, raw_tar_hrv, raw_tar_index, f_tar, f_tar_hrv, f_tar_index,
                        raw_pred, raw_pred_hrv, raw_pred_index, f_pred, f_pred_hrv, f_pred_index):
    # target
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 6))
    fig.suptitle('Target BPV signal HRV Analysis \n', fontweight='bold')
    ax[0].set_title('(a)')
    ax[0].plot(raw_tar.cpu().numpy(), '-.', color='gray', label='raw target')
    ax[0].plot(raw_tar_index[raw_tar_index >= 0].cpu().numpy(),
               raw_tar[raw_tar_index[raw_tar_index >= 0].cpu().numpy()].cpu().numpy(), 'b.',
               label='raw target peak')
    ax[0].plot(f_tar.cpu().numpy(), color='darkorange', label='filtered target')
    ax[0].plot(f_tar_index[f_tar_index >= 0].cpu().numpy(),
               f_tar[f_tar_index[f_tar_index >= 0].cpu().numpy()].cpu().numpy(), 'rx',
               label='filtered target peak')
    ax[0].set_xticks(np.arange(0, len(raw_tar) + 30, 30), np.arange(0, len(raw_tar) + 30, 30) // 30)
    ax[0].set_xlabel('Time (seconds)')
    ax[0].legend(loc='lower right')
    # prediction
    ax[1].set_title('(b)')
    ax[1].plot(raw_tar_hrv[raw_tar_hrv >= 0].cpu().numpy(), '-.', color='gray', label='raw target hrv')
    ax[1].plot(raw_tar_hrv[raw_tar_hrv >= 0].cpu().numpy(), 'b.')
    ax[1].plot(f_tar_hrv[f_tar_hrv >= 0].cpu().numpy(), color='darkorange', label='filtered target hrv')
    ax[1].plot(f_tar_hrv[f_tar_hrv >= 0].cpu().numpy(), 'rx')
    ax[1].set_xlabel('HRV Count')
    ax[1].set_xticks(np.arange(0, len(raw_tar_hrv[raw_tar_hrv >= 0]), 1))
    ax[1].set_ylabel('HRV (milliseconds)')
    ax[1].legend(loc='lower right')
    fig.tight_layout()
    plt.show()
    plt.close()

    # prediction
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 6))
    fig.suptitle('Predicted BVP signal HRV Analysis \n', fontweight='bold')
    ax[0].set_title('(a)')  # raw BVP HR: ' + str(int(raw_pred_hr)) + ' filtered BVP HR: ' + str(int(f_pred_hr)))
    ax[0].plot(raw_pred.cpu().numpy(), '-.', color='gray', label='raw prediction')
    ax[0].plot(raw_pred_index[raw_pred_index >= 0].cpu().numpy(),
               raw_pred[raw_pred_index[raw_pred_index >= 0].cpu().numpy()].cpu().numpy(), 'b.',
               label='raw prediction peak')
    ax[0].plot(f_pred.cpu().numpy(), color='darkorange', label='filtered prediction')
    ax[0].plot(f_pred_index[f_pred_index >= 0].cpu().numpy(),
               f_pred[f_pred_index[f_pred_index >= 0].cpu().numpy()].cpu().numpy(), 'rx',
               label='filtered prediction peak')
    ax[0].set_xticks(np.arange(0, len(raw_tar) + 30, 30), np.arange(0, len(raw_tar) + 30, 30) // 30)
    ax[0].set_xlabel('Time (seconds)')
    ax[0].legend(loc='lower right')
    # plt.subplot(2, 1, 2)
    ax[1].set_title('(b)')  # HRV Comparison' + '[ Peak score: ' + str(peak_score) + ']')
    ax[1].plot(raw_pred_hrv[raw_pred_hrv >= 0].cpu().numpy(), '-.', color='gray', label='raw prediction hrv')
    ax[1].plot(raw_pred_hrv[raw_pred_hrv >= 0].cpu().numpy(), 'b.')
    ax[1].plot(f_pred_hrv[f_pred_hrv >= 0].cpu().numpy(), color='darkorange', label='filtered prediction hrv')
    ax[1].plot(f_pred_hrv[f_pred_hrv >= 0].cpu().numpy(), 'rx')
    ax[1].set_xlabel('HRV Count')
    ax[1].set_xticks(np.arange(0, len(raw_pred_hrv[raw_pred_hrv >= 0]), 1))
    ax[1].set_ylabel('HRV (milliseconds)')
    plt.legend(loc='lower right')
    fig.tight_layout()
    plt.show()
    print('test')


def hr_comparison_bpf(hr_label_fft, hr_pred_fft, hr_pred_fft_filtered,
                      hr_label_peak, hr_pred_peak, hr_pred_peak_filtered):
    plt.title('(a)')
    plt.scatter(x=hr_label_fft.detach().cpu().numpy(),
                y=hr_pred_fft.detach().cpu().numpy(),
                color='blue', alpha=0.2, marker='2', label='FFT HR Prediction')
    plt.scatter(x=hr_label_fft.detach().cpu().numpy(),
                y=hr_pred_fft_filtered.detach().cpu().numpy(),
                color='red', alpha=0.2, marker='1', label='FFT HR Prediction Filtered')
    plt.xlim(40, 150)
    plt.xlabel('Target HR')
    plt.ylabel('Predicted HR')
    plt.legend(loc='upper left')
    plt.show()
    plt.title('(b)')
    plt.scatter(x=hr_label_peak[0].detach().cpu().numpy(),
                y=hr_pred_peak[0].detach().cpu().numpy(),
                color='blue', alpha=0.2, marker='2', label='Peak HR Prediction')
    plt.scatter(x=hr_label_peak[0].detach().cpu().numpy(),
                y=hr_pred_peak_filtered.detach().cpu().numpy(),
                color='red', alpha=0.2, marker='1', label='Peak HR Prediction Filtered')
    plt.legend(loc='upper left')
    plt.xlim(40, 150)
    plt.xlabel('Target HR')
    plt.ylabel('Predicted HR')
    plt.show()
