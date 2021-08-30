import numpy as np
from scipy.io import loadmat
from scipy.signal import find_peaks
import scipy
import pandas as pd
import heartpy as hp

def PPNet_preprocess_Mat(path):
    '''
    :param path: label file path
    :return: delta pulse
    '''

    data = loadmat(path)['p']

    #size = 125
    time = 2
    fs = 125
    interval = time * fs
    ppg = []
    sbp = []  # Systolic Blood Pressure
    dbp = []  # Diastolic Blood Pressue
    hr = []   # Heart Rate

    for i in range(1000):
        temp_mat = data[0, i]
        temp_length = temp_mat.shape[1]

        iteration = (int)((temp_length - 1000)/250 + 1)

        for j in range(iteration):
            temp_ppg = temp_mat[0,j * interval : j * interval + 1000]
            temp_bp = temp_mat[0, j * interval : j * interval + 1000]

            wd, m = hp.process(temp_ppg, sample_rate=fs)

            #ppg.append(temp_ppg[0:temp_ppg.size:4])
            # ppg.append(np.resize(temp_ppg,(temp_ppg.size/4)))
            # ppg.append(np.interp(np.arrange(0)))
            ppg.append(temp_ppg.reshape(-1, (temp_ppg.size/4)).mean(axis=1))
            sbp.append(max(temp_bp))
            dbp.append(min(temp_bp))
            hr.append(m['bpm'])


    return ppg, sbp, dbp, hr
