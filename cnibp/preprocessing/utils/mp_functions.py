import os
import time

import numpy as np

import multiprocessing as mp
import h5py
import pickle

import cnibp.preprocessing.utils.signal_utils as su


def get_process_num(target_num: int):
    """
    :param target_num: total number of task(segment) to be preprocessed
    :return: process_num: number of process to be used
    """

    divisors = []
    for i in range(1, int(target_num ** 0.5) + 1):
        if target_num % i == 0:
            divisors.append(i)
            if i != target_num // i:
                divisors.append(target_num // i)
    available_divisor = [x for x in divisors if x < os.cpu_count()]
    # if np.max(available_divisor) < os.cpu_count() // 2:
    #     process_num = os.cpu_count()
    # else:
    #     process_num = np.max(available_divisor)
    # if process_num < os.cpu_count():
    #     if process_num % 2 == 0:
    #         return process_num
    #     else:
    #         return process_num + 1
    # # if np.max(available_divisor) < os.cpu_count() // 2:
    # #     return os.cpu_count()
    # # else:
    # #     if np.max(available_divisor) % 2 == 0 and np.max(available_divisor) < os.cpu_count():
    # #         return np.max(available_divisor)
    # #     else:
    # #         return np.max(available_divisor) + 1

    return os.cpu_count() if np.max(available_divisor) < os.cpu_count() // 2 else np.max(available_divisor)


def multi_processing_sort_by_file_size(model_name, target_function, mode: str,
                                       parameters, dset_path: str, splitted_segments_by_f_size, patient_info_df):
    print(f'[{model_name} {mode} dataset]')
    print('dataset name : MIMIC-III')
    # print(f'number of segments: {len(splitted_segments_by_f_size)}')
    print(f'save to: {dset_path}')
    # split_by_size = []
    data_len_list = []
    acc_len_list = []

    # if get_process_num(len(segments)) > os.cpu_count():
    #     process_num = os.cpu_count()
    # else:
    #     process_num = get_process_num(len(segments))
    # process_num = 1
    process_num = 96
    print(f'number of processes: {process_num}')

    # ''' Model selection '''
    # if model_name == 'BPNet':
    #     sig_len, samp_rate = 750, 60
    # else:
    #     sig_len, samp_rate = 3000, 300

    print('Sorting data by file size...')
    # sorted_by_fsize = sorted(segments, key=lambda fs: os.stat(fs.replace('.hea', '.dat')).st_size)

    # if 'light0' in size_list:
    #     split_by_size.append(sorted_by_fsize[:int(len(sorted_by_fsize) * 0.25)])
    # if 'light1' in size_list:
    #     split_by_size.append(sorted_by_fsize[int(len(sorted_by_fsize) * 0.25):int(len(sorted_by_fsize) * 0.4)])
    # if 'light2' in size_list:
    #     split_by_size.append(sorted_by_fsize[int(len(sorted_by_fsize) * 0.4):int(len(sorted_by_fsize) * 0.55)])
    # if 'light3' in size_list:
    #     split_by_size.append(sorted_by_fsize[int(len(sorted_by_fsize) * 0.55):int(len(sorted_by_fsize) * 0.70)])
    # if 'heavy1' in size_list:
    #     split_by_size.append(sorted_by_fsize[int(len(sorted_by_fsize) * 0.70):int(len(sorted_by_fsize) * 0.85)])
    # if 'heavy2' in size_list:
    #     split_by_size.append(sorted_by_fsize[int(len(sorted_by_fsize) * 0.80):int(len(sorted_by_fsize) * 0.95)])
    # light0 = sorted_by_fsize[:int(len(sorted_by_fsize) * 0.25)]  # not used having no valid data
    # light1 = sorted_by_fsize[int(len(sorted_by_fsize) * 0.25):int(len(sorted_by_fsize) * 0.4)]
    # light2 = sorted_by_fsize[int(len(sorted_by_fsize) * 0.4):int(len(sorted_by_fsize) * 0.55)]
    # light3 = sorted_by_fsize[int(len(sorted_by_fsize) * 0.55):int(len(sorted_by_fsize) * 0.70)]  # htop best
    # heavy1 = sorted_by_fsize[int(len(sorted_by_fsize) * 0.70):int(len(sorted_by_fsize) * 0.85)]
    # heavy2 = sorted_by_fsize[int(len(sorted_by_fsize) * 0.80):int(len(sorted_by_fsize) * 0.95)]
    # heavy3 = sorted_by_fsize[int(len(sorted_by_fsize) * 0.95):] # eliminated due to long time consumption
    # split_by_size = [heavy1]  # for real data inspection
    # split_by_size = [light3]  # for debugging
    # split_by_size = [light2, light3]  # for fast test
    # split_by_size = [light0, light1]  #, light2, light3, heavy1, heavy2]  # for total data
    # split_by_size = {'light0': sorted_by_fsize[:int(len(sorted_by_fsize) * 0.25)]}

    # def multi_processing(sort_list, process_num, target_function)
    print('reading_total_data...')
    ple_tot = np.zeros((1, 750))
    ple_cyc = np.zeros((1, 100))
    abp_tot = np.zeros((1, 750))
    abp_cyc = np.zeros((1, 100))
    dbp_tot = np.zeros((1, 2, 15))
    sbp_tot = np.zeros((1, 2, 15))
    info_tot = np.zeros((1, 5))
    if parameters['mode'] == 'total':
        p_status_tot = np.zeros((1, 5))
        a_status_tot = np.zeros((1, 8))
    elif parameters['mode'] == 'none':
        p_status_tot = np.zeros((1, 3))
        a_status_tot = np.zeros((1, 5))
    elif parameters['mode'] == 'damp':
        p_status_tot = np.zeros((1, 3))
        a_status_tot = np.zeros((1, 6))
    else:
        p_status_tot = np.zeros((1, 4))
        a_status_tot = np.zeros((1, 6))

    # ple_tot = np.zeros((1, 3, 750))
    # size_tot = np.zeros((1, 2))
    # ohe_tot = np.zeros((1, 7))

    # invalid_ple_tot = np.zeros((1, 750))
    # invalid_abp_tot = np.zeros((1, 750))
    # eliminated_tot = np.zeros(7)
    '''get patient info'''

    for s in splitted_segments_by_f_size:
        segments_per_process = np.array_split(s, process_num)
        print(f'number of segments per process: {len(segments_per_process[0])}')
        with mp.Manager() as manager:
            start_time = time.time()

            info_total = manager.list()
            ple_total = manager.list()
            ple_cycle = manager.list()
            abp_total = manager.list()
            abp_cycle = manager.list()
            dbp_total = manager.list()
            sbp_total = manager.list()
            p_status_total = manager.list()
            a_status_total = manager.list()

            workers = [mp.Process(target=target_function,
                                  args=(process_i, segments_per_process[process_i],
                                        su.select_mode(parameters['mode']), parameters['chunk_size'],
                                        parameters['sampling_rate'],
                                        parameters['corr_threshold'], parameters['ple_scale'], parameters['hdf_flag'],
                                        patient_info_df, ple_cycle, abp_cycle,
                                        info_total, ple_total, abp_total, dbp_total, sbp_total,
                                        p_status_total, a_status_total,
                                        )) for process_i in range(process_num)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()

            print('--- %s seconds ---' % (time.time() - start_time))
            if len(info_total) != 0:
                info_tot = np.concatenate((info_tot, np.array(info_total, dtype=float)), axis=0)
                ple_tot = np.concatenate((ple_tot, np.array(ple_total)), axis=0)
                ple_cyc = np.concatenate((ple_cyc, np.array(ple_cycle)), axis=0)
                abp_tot = np.concatenate((abp_tot, np.array(abp_total)), axis=0)
                abp_cyc = np.concatenate((abp_cyc, np.array(abp_cycle)), axis=0)
                dbp_tot = np.concatenate((dbp_tot, np.array(dbp_total)), axis=0)
                sbp_tot = np.concatenate((sbp_tot, np.array(sbp_total)), axis=0)
                p_status_tot = np.concatenate((p_status_tot, np.array(p_status_total)), axis=0)
                a_status_tot = np.concatenate((a_status_tot, np.array(a_status_total)), axis=0)
                print('data added {}'.format(len(info_total)))
                data_len_list.append(len(info_total))

            else:
                print('no data added')
                data_len_list.append(len(info_total))

            manager.shutdown()

    data_len_list.append(len(info_tot) - 1)
    dset = h5py.File(
        dset_path + str(mode) + '_' + parameters['gender'] + '_' + str(parameters['corr_threshold']) + '.hdf5', 'w')

    # if gender == 0:
    #     dset = h5py.File(dset_path + str(dataset) + '_total_' + str(threshold) + '.hdf5', 'w')
    # elif gender == 1:
    #     dset = h5py.File(dset_path + str(dataset) + '_male_' + str(threshold) + '.hdf5', 'w')
    # else:
    #     dset = h5py.File(dset_path + str(dataset) + '_female_' + str(threshold) + '.hdf5', 'w')

    # dset['info'] = np.array(info_tot[1:], dtype='str')
    dset['info'] = info_tot[1:]
    dset['ple'] = ple_tot[1:]
    dset['ple_cycle'] = ple_cyc[1:]
    dset['abp'] = abp_tot[1:]
    dset['abp_cycle'] = abp_cyc[1:]
    dset['dbp'] = dbp_tot[1:]
    dset['sbp'] = sbp_tot[1:]
    dset['p_status'] = p_status_tot[1:]
    dset['a_status'] = a_status_tot[1:]

    dset.close()
    manager.shutdown()

    return data_len_list
