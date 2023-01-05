import os
import wfdb
import random

from tqdm import tqdm
import multiprocessing as mp
# from multiprocessing import Process, shared_memory, Semaphore
import numpy as np
import matplotlib.pyplot as plt
import vid2bp.preprocessing.utils.multi_processing as multi
# import vid2bp.preprocessing.utils.data_shuffler as ds
import time
import h5py
import datetime as dt
import vid2bp.preprocessing.utils.signal_utils as su
# import vid2bp.preprocessing.utils.sutemp as sutemp
import vid2bp.preprocessing.utils.math_module as mm

'''
signal_utils.py 로 이동해야 할 함수
'''
from scipy import signal


def derivative(x):
    deriv = np.append(x[1:], x[-1]) - x
    deriv[-1] = np.mean(deriv[-3:-2])
    return deriv


def channel_aggregator(x, dx, ddx):
    return np.concatenate((x, dx, ddx), axis=0)


def down_sampling(original_signal, fs: int = 125, target_fs: int = 60):
    '''
    :param original_signal: signal to be down-sampled
    :param fs: original sampling rate
    :param target_fs: target sampling rate
    :return:
    '''
    # if fs == target_fs:
    #     return original_signal
    # else:
    #     return signal.resample(original_signal, int(len(original_signal) * target_fs / fs))
    return signal.resample(original_signal, int(len(original_signal) * target_fs / fs))


def signal_QC(ple_chunk, abp_chunk):
    if np.isnan(ple_chunk).any() or np.isnan(abp_chunk).any() or \
            (np.var(abp_chunk) < 1) or \
            (not (np.sign(abp_chunk) > 0.0).all()):
        return False
    else:
        # plt.plot(abp_chunk)
        # plt.show()
        return True


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


def list_shuffler(path_list):
    """
    :param path_list: list of path to be shuffled
    :return: shuffled path_list
    """
    shuffle_cnt = random.randint(len(path_list), len(path_list) * 2)
    for c in tqdm(range(shuffle_cnt), desc='shuffling'):
        i = random.randint(0, len(path_list) - 1)
        j = random.randint(0, len(path_list) - 1)
        path_list[i], path_list[j] = path_list[j], path_list[i]
    return path_list


def get_segments_per_person(read_path: str):
    """
    :param read_path: path of a single patient (e.g. /hdd/hdd0/dataset/bpnet/adults/physionet.org/files/mimic3wdb/1.0/30/3001937_11)
    :return: all_file_paths: list of all segments of a single patient
    """
    all_file_paths = []
    for root, dirs, files in os.walk(read_path):
        for file in files:
            if file.endswith('.hea'):
                all_file_paths.append(root + '/' + file)
    all_file_paths = [p for p in all_file_paths if
                      p.endswith('.hea') and ('_' in p.split('/')[-1]) and ('layout' not in p)]
    random.shuffle(all_file_paths)
    return all_file_paths


def get_total_segment_path(read_path: str):
    """
    * if single patient have too many records,
        randomly select 5(or any number you like) records to prevent over-fitting to single patient.
    param: read_path: path of the raw dataset (e.g. /hdd/hdd0/dataset/bpnet/adults/physionet.org/files/mimic3wdb/1.0)
    return: all_shuffled_data_path: shuffled list of all patient's segments
    """
    total_patient_path = []
    train_shuffled_path, val_shuffled_path, test_shuffled_path = [], [], []

    # get all patient's path
    for root, dirs, files in os.walk(read_path):
        if len(files) > 0:
            total_patient_path.append(root)
    total_patient_path = [p for p in total_patient_path if '_' in p.split('/')[-1]]
    print('shuffling all patient path...')
    total_patient_path = list_shuffler(total_patient_path)

    # split train, val, test
    train_patient_num = int(len(total_patient_path) * 0.8)
    val_patient_num = int(len(total_patient_path) * 0.1)
    train_path = total_patient_path[:train_patient_num]
    val_path = total_patient_path[train_patient_num:train_patient_num + val_patient_num]
    test_path = total_patient_path[train_patient_num + val_patient_num:]

    # get train segments per patient
    print('get_train_segment_path...')
    for tr in tqdm(train_path, desc='Train_segment'):
        train_segments = get_segments_per_person(tr)
        if len(train_segments) > 5:
            reduced_train_segments = random.sample(train_segments, 5)
            train_shuffled_path.extend(reduced_train_segments)
        else:
            train_shuffled_path.extend(train_segments)
    # get val segments per patient
    print('get_validation_segment_path...')
    for v in tqdm(val_path, desc='Val_segment'):
        val_segments = get_segments_per_person(v)
        if len(val_segments) > 5:
            reduced_val_segments = random.sample(val_segments, 5)
            val_shuffled_path.extend(reduced_val_segments)
        else:
            val_shuffled_path.extend(val_segments)
    # get test segments per patient
    print('get_test_segment_path...')
    for te in tqdm(test_path, desc='Test_segment'):
        test_segments = get_segments_per_person(te)
        if len(test_segments) > 10:
            reduced_test_segments = random.sample(test_segments, 10)
            test_shuffled_path.extend(reduced_test_segments)
        else:
            test_shuffled_path.extend(test_segments)
    # for p in all_patient_paths:
    #     segments_per_person = get_segments_per_person(p)
    #     if len(segments_per_person) > 5:
    #         # 한 사람의 segment가 10개가 넘으면, 랜덤하게 10개를 뽑아서 가져옴
    #         reduced_segments_per_person = random.sample(list_shuffler(segments_per_person), 5)
    #         # random.shuffle(segments_per_person)
    #         # segments_per_person = segments_per_person[:10]
    #         all_shuffled_data_path.extend(reduced_segments_per_person)
    #     else:
    #         all_shuffled_data_path.extend(segments_per_person)
    # # 사람 순서대로 가져온 segments_per_person을 랜덤하게 다시 셔플
    # print('total number of patients: ', all_patient_num)
    # print('total number of segments: ', len(all_shuffled_data_path))
    # return list_shuffler(all_shuffled_data_path)
    # return all_shuffled_data_path
    print('total number of patients: ', len(total_patient_path))
    print('total number of train patients: ', len(train_path))
    print('total number of val patients: ', len(val_path))
    print('total number of test patients: ', len(test_path))
    print('total number of train segments: ', len(train_shuffled_path))
    print('total number of val segments: ', len(val_shuffled_path))
    print('total number of test segments: ', len(test_shuffled_path))
    return list_shuffler(train_shuffled_path), list_shuffler(val_shuffled_path), list_shuffler(test_shuffled_path)


def find_channel_idx(path):
    """
    param: path: path of a segment (e.g. /hdd/hdd0/dataset/bpnet/adults/physionet.org/files/mimic3wdb/1.0/30/3001937_11)
    return: idx: index of the ple, abp channel
    """
    record = wfdb.rdrecord(path)
    channel_names = record.sig_name
    ple_idx = [p for p in range(len(channel_names)) if channel_names[p] == 'PLETH'][0]
    abp_idx = [a for a in range(len(channel_names)) if channel_names[a] == 'ABP'][0]

    return ple_idx, abp_idx


# def read_record(path):


def read_total_data(id: int, segment_list: list, ple_total: list, abp_total: list, size_total: list, chunk_size: int,
                    sampling_rate: int):
    """

    * if single record is shorter than 6 seconds, skip it to consider only long enough to have respiratory cycles
       else, slice it into 6 seconds segments

    ** if single record is too long,
        5 consecutive chunks are selected to prevent over-fitting to single record.
        -> it is to include as many patients as possible in datasets


    param:
        path: path of a patient (e.g. /hdd/hdd0/dataset/bpnet/adults/physionet.org/files/mimic3wdb/1.0/30/3001937_11)
        sampfrom: start index of the segment
        sampto: end index of the segment
    return:
        record: wfdb record object containing PLETH and ABP signals
        patient_records: list of wfdb record
    """

    for segment in tqdm(segment_list, desc='process-' + str(id), leave=False):
        chunk_per_segment = 0
        segment = segment.strip('.hea')
        ple_idx, abp_idx = find_channel_idx(segment)
        ple, abp = np.squeeze(np.split(wfdb.rdrecord(segment, channels=[ple_idx, abp_idx]).p_signal, 2, axis=1))

        # check segment length if it is longer than 6 seconds
        if type(abp) != np.float64 and type(ple) != np.float64 and len(ple) > chunk_size and len(abp) > chunk_size:
            # select signal set only longer than 6 seconds, split into 6 seconds chunks
            ple_split = np.array_split(ple[:(len(ple) // chunk_size) * chunk_size], len(ple) // chunk_size)
            abp_split = np.array_split(abp[:(len(abp) // chunk_size) * chunk_size], len(abp) // chunk_size)
        else:
            continue

        for p, a in zip(ple_split, abp_split):
            flag, mean_dbp, mean_sbp, mean_map = su.signal_respiration_checker(a, p, threshold=0.9)
            # flag, mean_dbp, mean_sbp, mean_map = sutemp.signal_slicing(a, p)
            if flag:
                # ple = down_sampling(p, target_fs=sampling_rate)
                ple_total.append(mm.channel_cat(down_sampling(p, target_fs=sampling_rate)))
                abp_total.append(down_sampling(a, target_fs=sampling_rate))
                size_total.append([mean_dbp, mean_sbp, mean_map])
                chunk_per_segment += 1
            else:
                continue
            # else:
            #     continue
            if chunk_per_segment == 10:
                break


def multi_processing(model_name, dataset: str, total_segments):
    '''
    param:
        model_name: name of model to train (e.g. 'BPNet', 'UNet', 'LSTM'...)
        data_path: path of the dataset (e.g. /hdd/hdd0/dataset/bpnet/adults/physionet.org/files/mimic3wdb/1.0)
    return:
        None
    '''

    x = dt.datetime.now()
    dset_path = '/hdd/hdd1/dataset/bpnet/preprocessed_' + str(x.year) + str(x.month) + str(x.day) + '/'
    if not os.path.exists(dset_path):
        os.mkdir(dset_path)

    print(f'[{model_name} {dataset} dataset]')
    print('dataset name : MIMIC-III')
    print(f'number of segments: {len(total_segments)}')
    print(f'save to: {dset_path}')

    # process_num = get_process_num(len(total_segments))
    # if process_num % 2 != 0:
    #     process_num += 1
    process_num = 144
    print(f'number of processes: {process_num}')
    # processes = []

    ''' Model selection '''
    if model_name == 'BPNet':
        sig_len, samp_rate = 750, 60
    else:
        sig_len, samp_rate = 3000, 300

    ''' Multi-processing '''
    print('sorting data by size... ')
    '''
    size 
    30% 192, 58초 소요 len: 159
    50% 192, 123초 소요
    70% 192, 400초 소요
    80% 192, 905초 소요
    800 : 1
    400 : 3
    300 : 12
    200 : 77
    100 : 470

    '''
    sorted_by_fsize = sorted(total_segments, key=lambda s: os.stat(s.replace('.hea', '.dat')).st_size)
    # weight = [0.7, 0.1, 0.005, ]
    # light_segments = total_segments[int(len(total_segments)*0.4):int(len(total_segments)*0.70)]
    # heavy_segments = total_segments[int(len(total_segments)*0.70):]
    # segments_per_process = np.array_split(light_segments, process_num)
    # segments_per_process = np.array_split(heavy_segments, process_num)
    light0 = sorted_by_fsize[:int(len(sorted_by_fsize) * 0.25)]
    light1 = sorted_by_fsize[int(len(sorted_by_fsize) * 0.25):int(len(sorted_by_fsize) * 0.4)]
    light2 = sorted_by_fsize[int(len(sorted_by_fsize) * 0.4):int(len(sorted_by_fsize) * 0.55)]
    light3 = sorted_by_fsize[int(len(sorted_by_fsize) * 0.55):int(len(sorted_by_fsize) * 0.7)]
    # heavy1 = sorted_by_fsize[int(len(sorted_by_fsize) * 0.70):int(len(sorted_by_fsize) * 0.85)]
    # heavy2 = sorted_by_fsize[int(len(sorted_by_fsize) * 0.80):int(len(sorted_by_fsize) * 0.95)]
    # heavy3 = sorted_by_fsize[int(len(sorted_by_fsize) * 0.95):]
    # split_by_size = [light1, light2, heavy1, heavy2]
    split_by_size = [light1, light2, light3]
    print('reading_total_data...')
    # ple_tot, abp_tot, size_tot = [], [], []
    # ple_l1, abp_l1, size_l1 = [], [], []
    # ple_l2, abp_l2, size_l2 = [], [], []
    # ple_h1, abp_h1, size_h1 = [], [], []
    # ple_h2, abp_h2, size_h2 = [], [], []
    # ple_h3, abp_h3, size_h3 = [], [], []
    ple_tot = np.zeros((1, 3, 360))
    abp_tot = np.zeros((1, 360))
    size_tot = np.zeros((1, 3))
    for s in split_by_size:
        segments_per_process = np.array_split(s, process_num)
        print(f'number of segments per process: {len(segments_per_process[0])}')
        with mp.Manager() as manager:
            start_time = time.time()
            ple_total = manager.list()
            abp_total = manager.list()
            size_total = manager.list()
            workers = [mp.Process(target=read_total_data,
                                  args=(i, segments_per_process[i],
                                        ple_total, abp_total, size_total,
                                        sig_len, samp_rate)) for i in range(process_num)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()

            print('--- %s seconds ---' % (time.time() - start_time))
            # ple_total = np.array(ple_total)
            # abp_total = np.array(abp_total)
            # size_total = np.array(size_total)
            ple_tot = np.concatenate((ple_tot, np.array(ple_total)), axis=0)
            abp_tot = np.concatenate((abp_tot, np.array(abp_total)), axis=0)
            size_tot = np.concatenate((size_tot, np.array(size_total)), axis=0)
            # ple_temp = np.array(ple_total, dtype=object)
            # ple_tot.append(np.array(ple_total, dtype=object))
            # abp_tot.append(np.array(abp_total, dtype=object))
            # size_tot.append(np.array(size_total, dtype=object))

            # dset = h5py.File(dset_path + str(dataset) + '.hdf5', 'w')
            # dset['ple'] = ple_total
            # dset['abp'] = abp_total
            # dset['size'] = size_total
            # dset.close()
            manager.shutdown()

    dset = h5py.File(dset_path + str(dataset) + '.hdf5', 'w')
    # dset['ple'] = np.squeeze(np.array(ple_tot))
    dset['ple'] = ple_tot[1:]
    dset['abp'] = abp_tot[1:]
    dset['size'] = size_tot[1:]
    dset.close()

    print('total length: ', len(ple_tot))
    print(np.shape(ple_tot))
    print(np.shape(abp_tot))
    print(np.shape(size_tot))
    # print(np.shape(ple_total[0]))
    print(ple_tot[1][:100])
    print(abp_tot[1][:100])
    print(size_tot[1])


def dataset_split(model_name: str, data_path: str):
    print('dataset splitting...')
    train_segments, val_segments, test_segments = get_total_segment_path(data_path)
    multi_processing(model_name, 'train', train_segments)
    multi_processing(model_name, 'val', val_segments)
    multi_processing(model_name, 'test', test_segments)
    pass


dataset_split('BPNet', '/hdd/hdd1/dataset/bpnet/adults/physionet.org/files/mimic3wdb/1.0')
