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
    # print(f'save to: {dset_path}')
    data_len_list = []
    survived_ple_ratio_list = []
    survived_abp_ratio_list = []
    # if get_process_num(len(segments)) > os.cpu_count():
    #     process_num = os.cpu_count()
    # else:
    #     process_num = get_process_num(len(segments))
    process_num = 1
    # process_num = 48
    # print(f'number of processes: {process_num}')

    # ''' Model selection '''
    # if model_name == 'BPNet':
    #     sig_len, samp_rate = 750, 60
    # else:
    #     sig_len, samp_rate = 3000, 300

    print('Sorting data by file size...')

    print('reading_total_data...')
    ple_tot = np.zeros((1, parameters['chunk_size']))
    ple_cyc_len = np.zeros(1)
    ple_cyc = np.zeros((1, 100))
    abp_tot = np.zeros((1, parameters['chunk_size']))
    abp_cyc_len = np.zeros(1)
    abp_cyc = np.zeros((1, 100))
    dbp_tot = np.zeros((1, 2, 15))
    sbp_tot = np.zeros((1, 2, 15))
    info_tot = np.zeros((1, 6))
    if parameters['mode'] == 'total':
        p_status_tot = np.zeros((1, 4))
        a_status_tot = np.zeros((1, 8))
    elif parameters['mode'] == 'none':
        p_status_tot = np.zeros((1, 3))
        a_status_tot = np.zeros((1, 5))
    elif parameters['mode'] in ['underdamp', 'overdamp', 'flip']:
        p_status_tot = np.zeros((1, 3))
        a_status_tot = np.zeros((1, 6))
    else:  # for flat
        p_status_tot = np.zeros((1, 4))
        a_status_tot = np.zeros((1, 6))

    '''get patient info'''

    for s in splitted_segments_by_f_size:
        segments_per_process = np.array_split(s, process_num)
        # print(f'number of segments per process: {len(segments_per_process[0])}')
        with mp.Manager() as manager:
            start_time = time.time()

            info_total = manager.list()
            ple_total = manager.list()
            ple_cycle_len = manager.list()
            ple_cycle = manager.list()
            abp_total = manager.list()
            abp_cycle_len = manager.list()
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
                                        patient_info_df, ple_cycle, ple_cycle_len, abp_cycle, abp_cycle_len,
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
                ple_cyc_len = np.concatenate((ple_cyc_len, np.array(ple_cycle_len)), axis=0)
                ple_cyc = np.concatenate((ple_cyc, np.array(ple_cycle)), axis=0)
                abp_tot = np.concatenate((abp_tot, np.array(abp_total)), axis=0)
                abp_cyc_len = np.concatenate((abp_cyc_len, np.array(abp_cycle_len)), axis=0)
                abp_cyc = np.concatenate((abp_cyc, np.array(abp_cycle)), axis=0)
                dbp_tot = np.concatenate((dbp_tot, np.array(dbp_total)), axis=0)
                sbp_tot = np.concatenate((sbp_tot, np.array(sbp_total)), axis=0)
                p_status_tot = np.concatenate((p_status_tot, np.array(p_status_total)), axis=0)
                a_status_tot = np.concatenate((a_status_tot, np.array(a_status_total)), axis=0)
                # print('total inspected data: {}'.format(len(p_status_total) - 1))
                print('data added : {} / {}'.format(len(info_total), len(p_status_total)))
                # print('ple:', np.round(np.array(p_status_total).sum(axis=0) / len(p_status_total), 3))
                # print('abp:', np.round(np.array(a_status_total).sum(axis=0) / len(a_status_total), 3))
                data_len_list.append(len(info_total))
                # survived_ratio_list.append()

            else:
                print('no data added')
                # print('ple:', np.round(np.array(p_status_total).sum(axis=0) / len(p_status_total), 3))
                # print('abp:', np.round(np.array(a_status_total).sum(axis=0) / len(a_status_total), 3))
                data_len_list.append(len(info_total))

            manager.shutdown()

    data_len_list.append(len(info_tot) - 1)
    dset = h5py.File(
        dset_path + str(mode) + '_' + parameters['gender'] + '_' + str(parameters['corr_threshold']) + '.hdf5', 'w')

    analysis_dset = h5py.File(
        dset_path + str(mode) + '_' + parameters['gender'] + '_' + str(parameters['corr_threshold']) + '_status.hdf5',
        'w')

    # if gender == 0:
    #     dset = h5py.File(dset_path + str(dataset) + '_total_' + str(threshold) + '.hdf5', 'w')
    # elif gender == 1:
    #     dset = h5py.File(dset_path + str(dataset) + '_male_' + str(threshold) + '.hdf5', 'w')
    # else:
    #     dset = h5py.File(dset_path + str(dataset) + '_female_' + str(threshold) + '.hdf5', 'w')

    # dset['info'] = np.array(info_tot[1:], dtype='str')
    dset['ple'] = ple_tot[1:]
    dset['abp'] = abp_tot[1:]
    dset['dbp'] = dbp_tot[1:]
    dset['sbp'] = sbp_tot[1:]

    analysis_dset['info'] = info_tot[1:]
    analysis_dset['ple_cycle'] = ple_cyc[1:]
    analysis_dset['ple_cycle_len'] = ple_cyc_len[1:]
    analysis_dset['abp_cycle'] = abp_cyc[1:]
    analysis_dset['abp_cycle_len'] = abp_cyc_len[1:]
    analysis_dset['p_status'] = p_status_tot[1:]
    analysis_dset['a_status'] = a_status_tot[1:]
    survived_ple_ratio_list.append(np.round(np.append(np.array(p_status_tot).sum(axis=0) / len(p_status_tot),(len(p_status_tot))), 3))
    survived_abp_ratio_list.append(np.round(np.append(np.array(a_status_tot).sum(axis=0) / len(a_status_tot),(len(a_status_tot))), 3))
    dset.close()
    analysis_dset.close()
    manager.shutdown()

    return data_len_list, survived_ple_ratio_list, survived_abp_ratio_list
