from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

import h5py
import heartpy.peakdetection as hp_peak
from heartpy.datautils import rolling_mean
from heartpy.filtering import filter_signal

import vid2bp.preprocessing.signal_cleaner as cleaner


def arrange_by_index(data, index):
    arranged_data = []
    for target_data in data:
        arranged_target_data = []
        for i in range(len(index)):
            arranged_target_data.append(target_data[index[i]])
        arranged_data.append(arranged_target_data)
    return arranged_data


def preprocessing(original_data_path, save_path, mode):
    manager = multiprocessing.Manager()
    abp_total = manager.list()
    ple_total = manager.list()
    size_total = manager.list()
    eliminate_total = manager.list()

    eliminate_total.extend([0, 0, 0, 0, 0, 0])

    data_file = original_data_path + mode + ".hdf5"
    data_file = h5py.File(data_file, 'r')

    local_std_sbp = manager.list()
    local_std_sbp.append(0)
    original_data_size = len(data_file['abp'])
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
                                              data_file['size'][i * loop:],
                                              global_std_sbp, abp_total, ple_total, size_total, eliminate_total))
        else:
            p = multiprocessing.Process(target=additional_filter,
                                        args=(data_file['abp'][i * loop:(i + 1) * loop],
                                              data_file['ple'][i * loop:(i + 1) * loop],
                                              data_file['size'][i * loop:(i + 1) * loop],
                                              global_std_sbp, abp_total, ple_total, size_total, eliminate_total))

        p.start()
        process.append(p)

    for p in process:
        p.join()

    data_file_path = save_path + mode + ".hdf5"
    data_file = h5py.File(data_file_path, "w")

    data_file.create_dataset('ple', data=np.array(ple_total))
    data_file.create_dataset('abp', data=np.array(abp_total))
    data_file.create_dataset('size', data=np.array(size_total))

    data_file.close()
    print('Eliminated by SBP vs DBP: ', eliminate_total[0],'(',eliminate_total[0]/sum(eliminate_total)*100,'%)')
    print('Eliminated by PLE peak vs PLE bottom: ', eliminate_total[1], '(', eliminate_total[1] / sum(eliminate_total) * 100, '%)')
    print('Eliminated by non-peak signal: ', eliminate_total[2], '(', eliminate_total[2] / sum(eliminate_total) * 100, '%)')
    print('Eliminated by SBP vs PLE peak: ', eliminate_total[3], '(', eliminate_total[3] / sum(eliminate_total) * 100, '%)')
    print('Eliminated by DBP vs PLE bottom: ', eliminate_total[4], '(', eliminate_total[4] / sum(eliminate_total) * 100, '%)')
    print('Eliminated by SBP standard deviation: ', eliminate_total[5], '(', eliminate_total[5] / sum(eliminate_total) * 100, '%)')
    print('----------------------------------------------')
    print('Original data size: ', original_data_size)
    print('Eliminated total: ', sum(eliminate_total))
    print('Final data size: ', len(abp_total))
    print('Ratio of eliminated data: ', np.sum(eliminate_total) / original_data_size * 100, '%')


def additional_filter(ABP, PLE, SIZE, global_std_sbp, abp_total, ple_total, size_total, eliminate_total):
    # select normal range ABP
    normal_index = []
    for idx, target_abp in enumerate(ABP):
        if np.any(target_abp > 220) or np.any(target_abp < 40):
            continue
        else:
            normal_index.append(idx)

    ABP = [ABP[i] for i in normal_index]
    PLE = [PLE[i] for i in normal_index]
    SIZE = [SIZE[i] for i in normal_index]

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
    num_abp_bottom = [len(target_abp) for target_abp in bottom_abp]
    num_ple_bottom = [len(target_ple) for target_ple in bottom_ple]

    # find index where difference of abp peaks and abp bottoms are less than or equal to 2
    diff_index_abp = [i for i, (a, b) in enumerate(zip(num_abp_peak, num_abp_bottom)) if abs(a - b) <= 2]

    # number of eliminated signals
    eliminate_total[0] += len(ABP) - len(diff_index_abp)

    # arrange target signal by index
    ABP, PLE, SIZE, peak_abp, peak_ple, bottom_abp, bottom_ple, num_abp_peak, num_ple_peak, num_abp_bottom, num_ple_bottom \
        = arrange_by_index([ABP, PLE, SIZE, peak_abp, peak_ple, bottom_abp, bottom_ple, num_abp_peak, num_ple_peak, num_abp_bottom, num_ple_bottom], diff_index_abp)

    # find index where difference of peaks and bottoms are less than or equal to 2
    diff_index_ple = [i for i, (a, b) in enumerate(zip(num_ple_peak, num_ple_bottom)) if abs(a - b) <= 2]

    # number of eliminated signals
    eliminate_total[1] += len(ABP) - len(diff_index_ple)

    # arrange target signal by index
    ABP, PLE, SIZE, peak_abp, peak_ple, bottom_abp, bottom_ple, num_abp_peak, num_ple_peak, num_abp_bottom, num_ple_bottom \
        = arrange_by_index([ABP, PLE, SIZE, peak_abp, peak_ple, bottom_abp, bottom_ple, num_abp_peak, num_ple_peak, num_abp_bottom, num_ple_bottom], diff_index_ple)

    # find index where number of both peaks is more than or equal to 1
    positive_index = [i for i in range(len(num_abp_peak))
                      if num_abp_peak[i] >= 1 and num_ple_peak[i] >= 1
                      and num_abp_bottom[i] >= 1 and num_ple_bottom[i] >= 1]

    # number of eliminated signals
    eliminate_total[2] += len(ABP) - len(positive_index)

    # arrange target signals by positive index
    ABP, PLE, SIZE, peak_abp, peak_ple, denoised_abp, denoised_ple, num_abp_peak, num_ple_peak, num_abp_bottom, num_ple_bottom \
        = arrange_by_index([ABP, PLE, SIZE, peak_abp, peak_ple, denoised_abp, denoised_ple, num_abp_peak, num_ple_peak, num_abp_bottom, num_ple_bottom], positive_index)

    # compare number of peaks
    num_peak_diff = np.array(num_abp_peak) - np.array(num_ple_peak)
    num_peak_diff = np.abs(num_peak_diff)

    # find index where num_peak_diff is less than or equal to 2
    diff_index = np.where(num_peak_diff <= 2)[0]

    # number of eliminated signals
    eliminate_total[3] += len(ABP) - len(diff_index)

    # arrange target signals by diff_index
    ABP, PLE, SIZE, denoised_ple, denoised_abp, peak_abp, peak_ple, num_abp_peak, num_ple_peak, num_abp_bottom, num_ple_bottom \
        = arrange_by_index([ABP, PLE, SIZE, denoised_ple, denoised_abp, peak_abp, peak_ple, num_abp_peak, num_ple_peak, num_abp_bottom, num_ple_bottom], diff_index)

    # compare number of bottoms
    num_bottom_diff = np.array(num_abp_bottom) - np.array(num_ple_bottom)
    num_bottom_diff = np.abs(num_bottom_diff)

    # find index where num_bottom_diff is less than or equal to 2
    diff_index = np.where(num_bottom_diff <= 2)[0]

    # number of eliminated signals
    eliminate_total[4] += len(ABP) - len(diff_index)

    # arrange target signals by diff_index
    ABP, PLE, SIZE, denoised_ple, denoised_abp, peak_abp, peak_ple, num_abp_peak, num_ple_peak, num_abp_bottom, num_ple_bottom \
        = arrange_by_index([ABP, PLE, SIZE, denoised_ple, denoised_abp, peak_abp, peak_ple, num_abp_peak, num_ple_peak, num_abp_bottom, num_ple_bottom], diff_index)

    # find peak std
    peak_std = [np.std(target_abp[target_peak]) for target_abp, target_peak in zip(denoised_abp, peak_abp)]

    # find peak std where std is smaller than global std
    peak_std_index = np.where(np.array(peak_std) < global_std_sbp * 1.341)[0]

    # number of eliminated signals
    eliminate_total[5] += len(ABP) - len(peak_std_index)

    # arrange target signals by peak_std_index
    ABP, PLE, SIZE = arrange_by_index([ABP, PLE, SIZE], peak_std_index)

    abp_total.extend(ABP)
    ple_total.extend(PLE)
    size_total.extend(SIZE)


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
    original_data_path = "/home/najy/PycharmProjects/vid2bp_datasets/raw/"
    save_path = "/home/najy/PycharmProjects/vid2bp_datasets/vid2bp_additional_preprocessed/"

    for mode in ['train', 'val', 'test']:
        print('------ ' + mode + ' start ------')
        preprocessing(original_data_path, save_path, mode)
        print('------ ' + mode + ' end ------')
