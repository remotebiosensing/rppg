import heartpy as hp
import numpy as np
from scipy.io import loadmat
import scipy
import math


def PPNet_preprocess_Mat(path):
    '''
    :param path: label file path
    :return: delta pulse
    '''

    data = loadmat(path)['p']

    #size = 125
    time = 2
    fs = 125
    interval = time * fs # 250

    down_sample = 4

    ppg = []
    sbp = []  # Systolic Blood Pressure
    dbp = []  # Diastolic Blood Pressue
    hr = []   # Heart Rate

    for i in range(1000):
        temp_mat = data[0, i]

        max_bp = max(temp_mat[1])
        min_bp = min(temp_mat[1])

        if max_bp > 180 or max_bp < 80 :
            continue

        if min_bp > 130 or min_bp < 60 :
            continue

        try:
            wd, m = hp.process(temp_mat[0], sample_rate=fs,bpmmax=max_bp, bpmmin=min_bp)
            if m['bpm'] < 40 or m['bpm'] > 220:
                continue
        except hp.exceptions.BadSignalWarning:
            continue

        temp_mat_ppg = temp_mat[0]
        temp_mat_ppg -= np.mean(temp_mat_ppg)
        temp_mat_ppg /= np.std(temp_mat_ppg)

        temp_length = temp_mat.shape[1]

        iteration = (int)((temp_length - 1000)/250 + 1)

        ppg_tmp = []
        sbp_tmp = []
        dbp_tmp = []
        hr_tmp = []

        for j in range(iteration):
            temp_ppg = temp_mat_ppg[j * interval : j * interval + 1000]
            temp_bp = temp_mat[1,j * interval : j * interval + 1000]
            bpmax = max(temp_bp)
            bpmin = min(temp_bp)
            try:
                wd, m = hp.process(temp_ppg, sample_rate=fs,bpmmax=bpmax,bpmmin=bpmin)
            except hp.exceptions.BadSignalWarning:
                continue

            ppg_tmp.append(temp_ppg.reshape(-1, down_sample).mean(axis=1))
            sbp_tmp.append(max(temp_bp))
            dbp_tmp.append(min(temp_bp))
            hr_tmp.append(np.mean(wd['hr']))

        sbp_tmp -= np.mean(sbp_tmp)
        sbp_tmp /= np.std(sbp_tmp)

        dbp_tmp -= np.mean(dbp_tmp)
        dbp_tmp /= np.std(dbp_tmp)

        hr_tmp -= np.mean(hr_tmp)
        hr_tmp /= np.std(hr_tmp)

        ppg.extend(ppg_tmp)
        sbp.extend(sbp_tmp)
        dbp.extend(dbp_tmp)
        hr.extend(hr_tmp)


    return ppg, sbp, dbp, hr

def sig_to_BPfiltersig(label,N,fs,hf_low,hf_high):
    '''
    label = gt signal
    N = signal length = len(label)
    fs = fps
    hf_low = low cut off frequency
    hf_high = high cut off frequency
    '''

    # signal->fft signal
    fft_label = scipy.fft.fft(label)

    # fft signal->fft signal shift
    fft_freqs = scipy.fft.fftfreq(N, d=1/fs)

    # bandpass filter
    x_label = len(fft_freqs)
    tmp = fft_freqs[x_label // 2:]
    tmp2 = fft_freqs[:x_label // 2]
    fft_freqs = np.concatenate((tmp, tmp2))

    for i in range(len(fft_freqs)):
        if hf_low <= math.fabs(fft_freqs[i]) <= hf_high:
            continue
        else:
            fft_label[i] = 0

    #bp filter signal -> ifft
    ifft_bp_label = scipy.fft.ifft(fft_label)

    # for i in range(0, len(label), 32):
    #     plt.title('Comapre original signal and Filtered Time signal every 32frames')
    #     plt.plot(label[i:i + 32], label='Ground Truth', color='blue', linewidth=2, alpha=0.5)
    #     plt.plot(ifft_bp_label.real[i:i + 32], label='Filtered', color='red', linewidth=1)
    #     plt.legend()
    #     plt.grid()
    #     plt.show()

    return abs(ifft_bp_label)
