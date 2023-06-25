import random

import numpy as np
import torch

from rppg.models import get_model
from rppg.config import get_config
from rppg.dataset_loader import (dataset_loader, data_loader)
from rppg.preprocessing.dataset_preprocess import check_preprocessed_data
from rppg.utils.funcs import (MAE, RMSE, MAPE, corr)

import neurokit2 as nk
from scipy.sparse import spdiags
from scipy.signal import periodogram
from scipy.signal import butter
from scipy.signal import filtfilt
from scipy.signal import resample
import matplotlib.pyplot as plt

SEED = 0

# for Reproducible model
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # if use multi-GPU
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)

generator = torch.Generator()
generator.manual_seed(SEED)


def butter_bandpass(sig, lowcut, highcut, fs, order=2):
    # butterworth bandpass filter

    sig = np.reshape(sig, -1)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    y = filtfilt(b, a, sig)
    return y


def normalize(x):
    return (x - x.mean()) / x.std()


def detrend(signal, Lambda):
    """detrend(signal, Lambda) -> filtered_signal
    This function applies a detrending filter.
    This  is based on the following article "An advanced detrending method with application
    to HRV analysis". Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.
    *Parameters*
      ``signal`` (1d numpy array):
        The signal where you want to remove the trend.
      ``Lambda`` (int):
        The smoothing parameter.
    *Returns*
      ``filtered_signal`` (1d numpy array):
        The detrended signal.
    """
    signal_length = len(signal)

    # observation matrix
    H = np.identity(signal_length)

    # second-order difference matrix

    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot((H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), signal)
    return filtered_signal


def getCleanPPG(rawPPG, sampleRate=30, lowcut=0.6, highcut=4, model_type='DIFF'):
    if model_type == 'DIFF':
        cleanPPG = detrend(np.cumsum(rawPPG), 100)
    else:
        cleanPPG = detrend(rawPPG, 100)
    cleanPPG = butter_bandpass(cleanPPG, lowcut=lowcut, highcut=highcut, fs=sampleRate)
    return normalize(cleanPPG)


def createSignalDict(rawPPG, sampleRate=30, model_type='DIFF'):
    PPG_Clean = getCleanPPG(rawPPG, sampleRate=sampleRate, lowcut=0.6, highcut=4, model_type=model_type)
    HR_PSD, yValue_PSD, xAxis_PSD = fft_hr(PPG_Clean, fs=sampleRate)
    PPG_Peaks = nk.ppg_findpeaks(PPG_Clean, sampling_rate=sampleRate)['PPG_Peaks']
    # HRV = nk.signal_rate(PPG_Peaks, sampling_rate=sampleRate, desired_length=len(rawPPG))
    HRV = (60 / (np.diff(PPG_Peaks)/sampleRate))
    HRV = resample(HRV, len(rawPPG))
    HR_Peaks = HRV.mean()

    signalDict = {'PPG_Raw': normalize(rawPPG),
                  'SampleRate': sampleRate,
                  'PPG_Clean': PPG_Clean,
                  'HR_PSD': HR_PSD,
                  'Points_PSD': (xAxis_PSD, yValue_PSD),
                  'PPG_Peaks': PPG_Peaks,
                  'HRV': HRV,
                  'HR_Peaks': HR_Peaks}
    return signalDict


def plotPPGnHRV(signalDict):
    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(20, 8))
    ax1.set_xlabel("Time (seconds)")
    ax0.margins(0.01)
    ax1.margins(0.01)
    fig.suptitle("Photoplethysmogram (PPG)", fontweight="bold")

    plt.tight_layout(h_pad=1.4)
    # plt.tight_layout()

    sampling_rate = signalDict['SampleRate']
    x_axis = np.linspace(0, signalDict['PPG_Raw'].shape[0] / sampling_rate, signalDict['PPG_Raw'].shape[0])

    # Plot cleaned and raw PPG
    # ax0.set_title("Raw & Cleaned Signal")
    ax0.plot(x_axis, signalDict['PPG_Raw'], color="#B0BEC5", label="Raw", zorder=1)
    ax0.plot(x_axis, signalDict['PPG_Clean'], color="#FB1CF0", label="Clean", zorder=1, linewidth=1.5)
    # Plot peaks
    ax0.scatter(x_axis[signalDict['PPG_Peaks']], signalDict['PPG_Clean'][signalDict['PPG_Peaks']], color="#D60574",
                label="Peaks", zorder=2)
    ax0.legend(loc="upper right")

    # Rate
    ax1.set_title("Heart Rate")
    ax1.plot(x_axis, signalDict['HRV'], color="#FB661C", label="Rate", linewidth=1.5)
    ax1.axhline(y=signalDict['HR_Peaks'], label="Mean", linestyle="--", color="#FBB41C")
    ax1.legend(loc="upper right")

    return fig


def plotComparePPG(targetDict, predictionDict):
    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(20, 8))
    ax1.set_xlabel("Time (seconds)")
    ax0.margins(0.01)
    ax1.margins(0.01)
    fig.suptitle("", fontweight="bold")

    plt.tight_layout(h_pad=3)
    # plt.tight_layout()

    sampling_rate = targetDict['SampleRate']
    x_axis = np.linspace(0, targetDict['PPG_Raw'].shape[0] / sampling_rate, targetDict['PPG_Raw'].shape[0])

    # Plot Ground Truth PPG Signal
    ax0.set_title("Ground Truth Signal", fontweight="bold")
    ax0.plot(x_axis, targetDict['PPG_Raw'], color="#B0BEC5", label="Raw", zorder=1)
    ax0.plot(x_axis, targetDict['PPG_Clean'], color="#FB1CF0", label="Clean", zorder=1, linewidth=1.5)
    # Plot peaks
    ax0.scatter(x_axis[targetDict['PPG_Peaks']], targetDict['PPG_Clean'][targetDict['PPG_Peaks']], color="#D60574",
                label="Peaks", zorder=2)
    ax0.legend(loc="upper right")

    # Plot Prediction PPG Signal
    ax1.set_title("Prediction Signal", fontweight="bold")
    ax1.plot(x_axis, predictionDict['PPG_Raw'], color="#B0BEC5", label="Raw", zorder=1)
    ax1.plot(x_axis, predictionDict['PPG_Clean'], color="#FB1CF0", label="Clean", zorder=1, linewidth=1.5)
    # Plot peaks
    ax1.scatter(x_axis[predictionDict['PPG_Peaks']], predictionDict['PPG_Clean'][predictionDict['PPG_Peaks']],
                color="#D60574", label="Peaks", zorder=2)
    ax1.legend(loc="upper right")

    return fig


def plotCompareHRV(targetDict, predictionDict):
    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(20, 8))
    ax1.set_xlabel("Time (seconds)")
    ax0.margins(0.01)
    ax1.margins(0.01)
    fig.suptitle("", fontweight="bold")

    plt.tight_layout(h_pad=3)
    # plt.tight_layout()

    sampling_rate = targetDict['SampleRate']
    x_axis = np.linspace(0, targetDict['PPG_Raw'].shape[0] / sampling_rate, targetDict['PPG_Raw'].shape[0])

    # Plot Ground Truth PPG Signal
    ax0.set_title("Ground Truth Heart Rate", fontweight="bold")
    ax0.plot(x_axis, targetDict['HRV'], color="#FB661C", label="Rate", linewidth=1.5)
    ax0.axhline(y=targetDict['HR_Peaks'], label="Mean", linestyle="--", color="#FBB41C")
    ax0.legend(loc="upper right")

    # Plot Prediction PPG Signal
    ax1.set_title("Prediction Heart Rate", fontweight="bold")
    ax1.plot(x_axis, predictionDict['HRV'], color="#FB661C", label="Rate", linewidth=1.5)
    ax1.axhline(y=predictionDict['HR_Peaks'], label="Mean", linestyle="--", color="#FBB41C")
    ax1.legend(loc="upper right")

    return fig


def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def fft_hr(rawPPG, fs=30, lowcut=0.6, highcut=4):
    ppg_signal = np.expand_dims(rawPPG, 0)
    N = _next_power_of_2(ppg_signal.shape[1])
    f_ppg, pxx_ppg = periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= lowcut) & (f_ppg <= highcut))
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
    return hr, pxx_ppg, f_ppg



def eval_fn(eval_model, dataloaders, model_name, cal_type, metrics, eval_time_length):
    step = "Test"

    if model_name in ["DeepPhys", "TSCAN", "MTTS", "BigSmall", "EfficientPhys"]:
        model_type = 'DIFF'
    else:
        model_type = 'CONT'

    eval_model.eval()

    p = []
    t = []

    fs = 30
    time = eval_time_length

    interval = fs * time

    for dataloader in dataloaders:
        sub_pred = []
        sub_target = []
        for inputs, target in dataloader:
            sub_pred.extend(np.reshape(eval_model(inputs).cpu().detach().numpy(), (-1,)))
            sub_target.extend(np.reshape(target.cpu().detach().numpy(), (-1,)))

        remind = len(sub_pred) % interval
        if remind > 0:
            sub_pred = sub_pred[:-remind]
            sub_target = sub_target[:-remind]
        p.append(np.reshape(np.asarray(sub_pred), (-1, interval)))
        t.append(np.reshape(np.asarray(sub_target), (-1, interval)))
    return p, t



if __name__ == "__main__":
    fit_cfg = get_config("configs/fit.yaml")
    preprocess_cfg = get_config("configs/preprocess.yaml")

    check_preprocessed_data(fit_cfg, preprocess_cfg)

    datasets = dataset_loader(save_root_path=preprocess_cfg.dataset_path, fit_cfg=fit_cfg.fit)

    data_loaders = data_loader(datasets=datasets, fit_cfg=fit_cfg.fit)

    model = get_model(fit_cfg.fit)

    save_model_path = "/home/jh/PycharmProjects/rppg/rppg/DeepPhys/trainUBFC_testPURE_imgsize72_testlen0.pt"
    model.load_state_dict(torch.load(save_model_path))

    p, t = eval_fn(eval_model=model, dataloaders=data_loaders[0], model_name=fit_cfg.fit.model,
                   cal_type=fit_cfg.fit.test.cal_type,
                   metrics=fit_cfg.fit.test.metric, eval_time_length=fit_cfg.fit.test.eval_time_length)

    HR_fft_pred = []
    HR_fft_target = []
    HR_peaks_pred = []
    HR_peaks_target = []
    for sub_p, sub_t in zip(p, t):
        for clip_p, clip_t in zip(sub_p, sub_t):
            target_bvpDict = createSignalDict(clip_p, sampleRate=30, model_type=fit_cfg.fit.type)
            pred_bvpDict = createSignalDict(clip_t, sampleRate=30, model_type=fit_cfg.fit.type)

            HR_fft_pred.append(pred_bvpDict['HR_PSD'])
            HR_fft_target.append(target_bvpDict['HR_PSD'])

            HR_peaks_pred.append(pred_bvpDict['HR_Peaks'])
            HR_peaks_target.append(target_bvpDict['HR_Peaks'])

    HR_fft_pred = np.asarray(HR_fft_pred)
    HR_fft_target = np.asarray(HR_fft_target)
    HR_peaks_pred = np.asarray(HR_peaks_pred)
    HR_peaks_target = np.asarray(HR_peaks_target)
    idxList = np.isfinite(HR_peaks_target) & np.isfinite(HR_peaks_pred)
    HR_peaks_pred = HR_peaks_pred[idxList]
    HR_peaks_target = HR_peaks_target[idxList]

    test_result = {"FFT": {}, "Peaks": {}}
    test_result['FFT']['MAE'] = MAE(HR_fft_pred, HR_fft_target)
    test_result['Peaks']['MAE'] = MAE(HR_peaks_pred, HR_peaks_target)
    print("MAE(FFT)", test_result['FFT']['MAE'])
    print("MAE(Peaks)", test_result['Peaks']['MAE'])

    test_result['FFT']['RMSE'] = RMSE(HR_fft_pred, HR_fft_target)
    test_result['Peaks']['RMSE'] = RMSE(HR_peaks_pred, HR_peaks_target)
    print("RMSE(FFT)", test_result['FFT']['RMSE'])
    print("RMSE(Peaks)", test_result['Peaks']['RMSE'])

    test_result['FFT']['MAPE'] = MAPE(HR_fft_pred, HR_fft_target)
    test_result['Peaks']['MAPE'] = MAPE(HR_peaks_pred, HR_peaks_target)
    print("MAPE(FFT)", test_result['FFT']['MAPE'])
    print("MAPE(Peaks)", test_result['Peaks']['MAPE'])

    test_result['FFT']['corr'] = corr(HR_fft_pred, HR_fft_target)[0][1]
    test_result['Peaks']['corr'] = corr(HR_peaks_pred, HR_peaks_target)[0][1]
    print("corr(FFT)", test_result['FFT']['corr'])
    print("corr(Peaks)", test_result['Peaks']['corr'])



    plt.title(f"HR Estimation - FFT\n Correlation : {test_result['FFT']['corr']}")
    plt.scatter(HR_fft_target, HR_fft_pred, color="#D60574")
    plt.plot(range(0, 150), range(0, 150), '--', color='k')
    plt.ylabel('Ground Truth (bpm)')
    plt.xlabel('Prediction (bpm)')
    plt.xlim((0, 150))
    plt.ylim((0, 150))
    plt.grid()
    plt.show()

    plt.title(f"HR Estimation - Peaks\n Correlation : {test_result['Peaks']['corr']}")
    plt.scatter(HR_peaks_pred, HR_peaks_pred, color="#D60574")
    plt.plot(range(0, 150), range(0, 150), '--', color='k')
    plt.ylabel('Ground Truth (bpm)')
    plt.xlabel('Prediction (bpm)')
    plt.xlim((0, 150))
    plt.ylim((0, 150))
    plt.grid()
    plt.show()

    for sub_p, sub_t in zip(p, t):
        sub_p = sub_p.reshape(-1,)
        sub_t = sub_t.reshape(-1,)
        target_bvpDict = createSignalDict(sub_p, sampleRate=30, model_type=fit_cfg.fit.type)
        pred_bvpDict = createSignalDict(sub_t, sampleRate=30, model_type=fit_cfg.fit.type)

        # plotPPGnHRV(target_bvpDict)
        # plt.show()
        # plotPPGnHRV(pred_bvpDict)
        # plt.show()
        # plotComparePPG(target_bvpDict, pred_bvpDict)
        # plt.show()
        # plotCompareHRV(target_bvpDict, pred_bvpDict)
        # plt.show()
        # plt.savefig(f'/home/jh/Desktop/HR_debug/{dataPath[24:-3]}.png')
        # plt.show()

    print("END")
