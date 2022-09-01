import numpy as np
from scipy import signal
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

'''
num : 
1800 >> sampling frequency :30
3600 >> sampling frequency :60
'''


def downsampling(in_signal, num=1800):
    rst_sig = signal.resample(in_signal, num * 10)
    return rst_sig


def peak_detection(in_signal):
    # TODO SBP, DBP 구해야 함
    peak_list = []
    avg_peak, min_peak, max_peak = 0, 0, 0

    avg_peak = np.mean(peak_list)
    min_peak = np.min(peak_list)
    max_peak = np.max(peak_list)
    print('avg_peak :', avg_peak, 'min_peak :', min_peak, 'max_peak :', max_peak)

    return peak_list
