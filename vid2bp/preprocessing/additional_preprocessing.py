from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

import h5py
import heartpy.peakdetection as hp_peak
from heartpy.datautils import rolling_mean
from heartpy.filtering import filter_signal

import vid2bp.preprocessing.signal_cleaner as cleaner


def preprocessing(original_data_path, save_path, mode):
    manager = multiprocessing.Manager()
    abp_total = manager.list()
    ple_total = manager.list()

    data_file = original_data_path + mode + ".hdf5"
    data_file = h5py.File(data_file, 'r')

    local_std_sbp = manager.list()
    local_std_sbp.append(0)

    # multiprocessing
    num_cpu = multiprocessing.cpu_count()
    loop = int(len(data_file['abp']) / num_cpu)
    process_std = []
    for i in tqdm(range(num_cpu), desc='Calculating local std'):
        if i == num_cpu - 1:
            p_std = multiprocessing.Process(target=global_std_sbp_calculator,
                                            args=(data_file['abp'][i * loop:], local_std_sbp))
        else:
            p_std = multiprocessing.Process(target=global_std_sbp_calculator,
                                            args=(data_file['abp'][i * loop:(i + 1) * loop], local_std_sbp))
        process_std.append(p_std)
        p_std.start()

    for p_std in process_std:
        p_std.join()

    global_std_sbp = local_std_sbp[0] / num_cpu

    process = []

    for i in tqdm(range(num_cpu), desc='Preprocessing'):
        if i == num_cpu - 1:
            p = multiprocessing.Process(target=additional_filter,
                                        args=(data_file['abp'][i * loop:],
                                              data_file['ple'][i * loop:],
                                              global_std_sbp, abp_total, ple_total))
        else:
            p = multiprocessing.Process(target=additional_filter,
                                        args=(data_file['abp'][i * loop:(i + 1) * loop],
                                              data_file['ple'][i * loop:(i + 1) * loop],
                                              global_std_sbp, abp_total, ple_total))

        p.start()
        process.append(p)

    for p in process:
        p.join()

    data_file_path = save_path + mode + ".hdf5"
    data_file = h5py.File(data_file_path, "w")

    data_file.create_dataset('ple', data=np.array(ple_total))
    data_file.create_dataset('abp', data=np.array(abp_total))

    data_file.close()


def additional_filter(ABP, PLE, global_std_sbp, abp_total, ple_total):
    # select normal range ABP
    normal_index = []
    for idx, target_abp in enumerate(ABP):
        if np.any(target_abp > 220) or np.any(target_abp < 40):
            continue
        else:
            normal_index.append(idx)

    ABP = [ABP[i] for i in normal_index]
    PLE = [PLE[i] for i in normal_index]

    # denoise signals
    hf = 8
    fs = 125
    denoised_abp = [filter_signal(target_abp, cutoff=hf, sample_rate=fs, order=2, filtertype='lowpass') for
                    target_abp in ABP]
    denoised_ple = [target_ple[0] for target_ple in PLE]
    denoised_ple = [filter_signal(target_ple, cutoff=hf, sample_rate=fs, order=2, filtertype='lowpass') for
                    target_ple in denoised_ple]

    # find peak index
    rolling_sec = 1.5
    peak_abp = [cleaner.peak_detector(target_abp, rolling_sec, fs) for target_abp in denoised_abp]
    peak_ple = [cleaner.peak_detector(target_ple, rolling_sec, fs) for target_ple in denoised_ple]
    bottom_abp = [cleaner.bottom_detector(target_abp, rolling_sec, fs) for target_abp in denoised_abp]
    bottom_ple = [cleaner.bottom_detector(target_ple, rolling_sec, fs) for target_ple in denoised_ple]

    # arrange peak index
    arranged_peak_abp = []
    arranged_peak_ple = []
    arranged_bottom_abp = []
    arranged_bottom_ple = []
    for target_signal, target_peak, target_bottom in zip(denoised_abp, peak_abp, bottom_abp):
        arranged_peak = cleaner.SBP_DBP_arranger(target_signal, target_peak, target_bottom)
        arranged_peak_abp.append(arranged_peak[0])
        arranged_bottom_abp.append(arranged_peak[1])

    for target_signal, target_peak, target_bottom in zip(denoised_ple, peak_ple, bottom_ple):
        arranged_peak = cleaner.SBP_DBP_arranger(target_signal, target_peak, target_bottom)
        arranged_peak_ple.append(arranged_peak[0])
        arranged_bottom_ple.append(arranged_peak[1])

    peak_abp = arranged_peak_abp
    peak_ple = arranged_peak_ple
    bottom_abp = arranged_bottom_abp
    bottom_ple = arranged_bottom_ple

    # find number of peaks
    num_abp_peak = [len(target_abp) for target_abp in peak_abp]
    num_ple_peak = [len(target_ple) for target_ple in peak_ple]

    # find index where number of both peaks is more than or equal to 1
    positive_index = [i for i in range(len(num_abp_peak)) if num_abp_peak[i] >= 1 and num_ple_peak[i] >= 1]

    # 1. arrange target signals by positive index
    ABP = [ABP[i] for i in positive_index]
    PLE = [PLE[i] for i in positive_index]
    peak_abp = [peak_abp[i] for i in positive_index]
    peak_ple = [peak_ple[i] for i in positive_index]
    denoised_abp = [denoised_abp[i] for i in positive_index]
    denoised_ple = [denoised_ple[i] for i in positive_index]
    num_abp_peak = [num_abp_peak[i] for i in positive_index]
    num_ple_peak = [num_ple_peak[i] for i in positive_index]

    # compare number of peaks
    num_peak_diff = np.array(num_abp_peak) - np.array(num_ple_peak)
    num_peak_diff = np.abs(num_peak_diff)

    # find index where num_peak_diff is less than or equal to 2
    diff_index = np.where(num_peak_diff <= 2)[0]

    # 2. arrange target signals by diff_index
    ABP = [ABP[i] for i in diff_index]
    PLE = [PLE[i] for i in diff_index]
    denoised_ple = [denoised_ple[i] for i in diff_index]
    denoised_abp = [denoised_abp[i] for i in diff_index]
    peak_abp = [peak_abp[i] for i in diff_index]
    peak_ple = [peak_ple[i] for i in diff_index]

    # find peak std
    peak_std = [np.std(target_abp[target_peak]) for target_abp, target_peak in zip(denoised_abp, peak_abp)]

    # find peak std where std is smaller than global std
    peak_std_index = np.where(np.array(peak_std) < global_std_sbp)[0]
    # peak_std_index = [np.where(target_peak_std <= global_std_sbp * 1.341)[0] for target_peak_std in peak_std]

    # 3. arrange target signals by peak_std_index
    ABP = [ABP[i] for i in peak_std_index]
    PLE = [PLE[i] for i in peak_std_index]
    denoised_ple = [denoised_ple[i] for i in peak_std_index]
    denoised_abp = [denoised_abp[i] for i in peak_std_index]
    peak_abp = [peak_abp[i] for i in peak_std_index]
    peak_ple = [peak_ple[i] for i in peak_std_index]

    # plt.plot((denoised_abp[0] - np.min(denoised_abp[0])) / (np.max(denoised_abp[0]) - np.min(denoised_abp[0])))
    # plt.plot((denoised_ple[0] - np.min(denoised_ple[0])) / (np.max(denoised_ple[0]) - np.min(denoised_ple[0])))
    # plt.legend()
    # plt.show()

    abp_total.extend(ABP)
    ple_total.extend(PLE)


def global_std_sbp_calculator(ABP, global_sbp):
    # denoise signals
    hf = 8
    fs = 125
    denoised_abp = [filter_signal(target_abp, cutoff=hf, sample_rate=fs, order=2, filtertype='lowpass') for
                    target_abp in ABP]

    # find peak index
    rolling_sec = 1.5
    peak_abp = [cleaner.peak_detector(target_abp, rolling_sec, fs) for target_abp in denoised_abp]

    # calculate average peak std
    average_peak_std = []
    for target_abp, target_peak in zip(denoised_abp, peak_abp):
        if len(target_peak) != 0:
            average_peak_std.append(np.std(target_abp[target_peak]))
    average_peak_std = np.mean(average_peak_std)

    global_sbp[0] += average_peak_std


if __name__ == '__main__':
    original_data_path = "/home/najy/PycharmProjects/PPG2ABP_datasets/raw/"
    save_path = "/home/najy/PycharmProjects/vid2bp_datasets/vid2bp_additional_preprocessed/"

    for mode in ['train', 'val', 'test']:
        preprocessing(original_data_path, save_path, mode)
