import matplotlib.pyplot as plt
import rppg.utils.funcs
import numpy as np


def plot(raw_pred, raw_pred_hr, raw_pred_hrv, raw_pred_index, f_pred, f_pred_hr, f_pred_hrv, f_pred_index,
         raw_tar, raw_tar_hr, raw_tar_hrv, raw_tar_index, f_tar, f_tar_hr, f_tar_hrv, f_tar_index):
    # plt.title(str(model_name) + ' HRV Comparison: ' + str(eval_time_length) + ' seconds\n')
    # target
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.title('Target HRV Analysis \n\nraw HR: ' + str(raw_tar_hr) + ' filtered HR: ' + str(f_tar_hr))
    plt.plot(raw_tar.cpu().numpy(), '-.', label='raw target')
    plt.plot(raw_tar_index[raw_tar_index.nonzero().squeeze()].cpu().numpy(),
             raw_tar[raw_tar_index[raw_tar_index.nonzero().squeeze()].cpu().numpy()].cpu().numpy(), 'rx',
             label='raw target peak')
    plt.plot(f_tar.cpu().numpy(), label='filtered target')
    plt.plot(f_tar_index[f_tar_index.nonzero().squeeze()].cpu().numpy(),
             f_tar[f_tar_index[f_tar_index.nonzero().squeeze()].cpu().numpy()].cpu().numpy(), 'b.',
             label='filtered target peak')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.title('temp')
    plt.plot(raw_tar_hrv[raw_tar_hrv.nonzero().squeeze()].cpu().numpy(), '-.', label='raw target hrv')
    plt.plot(f_tar_hrv[f_tar_hrv.nonzero().squeeze()].cpu().numpy(), label='filtered target hrv')
    plt.semilogy(base=10)
    plt.legend()
    plt.show()
    plt.close()
    # prediction
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    peak_score = str(round(len(raw_pred_hrv[raw_pred_hrv.nonzero().squeeze()]) /
                           len(raw_tar_hrv[raw_tar_hrv.nonzero().squeeze()]), 3))
    plt.title('Prediction HRV Analysis \n\nraw HR: ' + str(raw_pred_hr) + ' filtered HR: ' + str(f_pred_hr))
    plt.plot(raw_pred.cpu().numpy(), '-.', label='raw target')
    plt.plot(raw_pred_index[raw_pred_index.nonzero().squeeze()].cpu().numpy(),
             raw_pred[raw_pred_index[raw_pred_index.nonzero().squeeze()].cpu().numpy()].cpu().numpy(), 'rx',
             label='raw target peak')
    plt.plot(f_pred.cpu().numpy(), label='filtered target')
    plt.plot(f_pred_index[f_pred_index.nonzero().squeeze()].cpu().numpy(),
             f_pred[f_pred_index[f_pred_index.nonzero().squeeze()].cpu().numpy()].cpu().numpy(), 'b.',
             label='filtered target peak')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.title('HRV Comparison' + '[ Peak score: ' + peak_score + ']')
    plt.plot(raw_pred_hrv[raw_pred_hrv.nonzero().squeeze()].cpu().numpy(), '-.', label='raw target hrv')
    plt.plot(f_pred_hrv[f_pred_hrv.nonzero().squeeze()].cpu().numpy(), label='filtered target hrv')
    plt.semilogy(base=10)
    plt.legend()
    plt.show()
    print('test')
