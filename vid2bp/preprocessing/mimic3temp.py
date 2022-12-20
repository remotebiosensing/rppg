import os
import wfdb
import random

from tqdm import tqdm
import multiprocessing as mp
# from multiprocessing import Process, shared_memory, Semaphore
import numpy as np
import vid2bp.preprocessing.utils.signal_utils as su
import matplotlib.pyplot as plt
import vid2bp.preprocessing.utils.multi_processing as multi
# import vid2bp.preprocessing.utils.data_shuffler as ds
import time
import h5py
import datetime as dt
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
    if np.max(available_divisor) < os.cpu_count() // 2:
        process_num = os.cpu_count()
    else:
        process_num = np.max(available_divisor)
    if process_num < os.cpu_count():
        if process_num % 2 == 0:
            return process_num
        else:
            return process_num + 1
    # if np.max(available_divisor) < os.cpu_count() // 2:
    #     return os.cpu_count()
    # else:
    #     if np.max(available_divisor) % 2 == 0 and np.max(available_divisor) < os.cpu_count():
    #         return np.max(available_divisor)
    #     else:
    #         return np.max(available_divisor) + 1

    # return os.cpu_count() if np.max(available_divisor) < os.cpu_count() // 2 else np.max(available_divisor)


def list_shuffler(path_list):
    """
    :param path_list: list of path to be shuffled
    :return: shuffled path_list
    """
    shuffle_cnt = random.randint(len(path_list), len(path_list) * 2)
    for c in range(shuffle_cnt):
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
        randomly select 10(or any number you like) records to prevent over-fitting to single patient.
    param: read_path: path of the raw dataset (e.g. /hdd/hdd0/dataset/bpnet/adults/physionet.org/files/mimic3wdb/1.0)
    return: all_shuffled_data_path: shuffled list of all patient's segments
    """
    all_patient_num = 0
    all_patient_paths = []
    all_shuffled_data_path = []
    # 사람별로 가져오는 작업(all_patient_paths), 1700개 segment 있는 사람도 있고, 1개 있는 사람도 있고
    for root, dirs, files in os.walk(read_path):
        if len(files) > 0:
            all_patient_num += 1
            all_patient_paths.append(root)
    all_patient_paths = [p for p in all_patient_paths if '_' in p.split('/')[-1]]
    # 가져온 사람 p를 돌면서 p 안에 있는 segment를 최대 35개 가져옴
    for p in all_patient_paths:
        segments_per_person = get_segments_per_person(p)
        if len(segments_per_person) > 5:
            # 한 사람의 segment가 10개가 넘으면, 랜덤하게 10개를 뽑아서 가져옴
            reduced_segments_per_person = random.sample(list_shuffler(segments_per_person), 5)
            # random.shuffle(segments_per_person)
            # segments_per_person = segments_per_person[:10]
            all_shuffled_data_path.extend(reduced_segments_per_person)
        else:
            all_shuffled_data_path.extend(segments_per_person)
    # 사람 순서대로 가져온 segments_per_person을 랜덤하게 다시 셔플
    print('total number of patients: ', all_patient_num)
    print('total number of segments: ', len(all_shuffled_data_path))
    return list_shuffler(all_shuffled_data_path)


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


def read_total_data(segment_list: list, ple_total: list, abp_total: list, chunk_size: int, sampling_rate: int):
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

    for segment in tqdm(segment_list):
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
            if signal_QC(p, a):  # signal integrity check for both ple and abp
                if su.signal_respiration_checker(a, p, threshold=0.95):
                    # ple_total.append(channel_aggregator(p, derivative(p), derivative(mm.diff_np(p))))
                    ple_total.append(down_sampling(p, target_fs=sampling_rate))
                    abp_total.append(down_sampling(a, target_fs=sampling_rate))
                    chunk_per_segment += 1
                else:
                    continue
            else:
                continue
            if chunk_per_segment == 5:
                break

    '''
    아니면 여기서 전체를 다시 순회하면서 preprocessing을 할 것인지장
    multi_processing에서 h5py 파일 저
    '''


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

    print(f'{model_name} {dataset} dataset')
    print('dataset name : MIMIC-III')
    print(f'number of segments: {len(total_segments)}')
    print(f'save to: {dset_path}')

    start_time = time.time()

    manager = mp.Manager()
    ple_total = manager.list()
    abp_total = manager.list()

    # process_num = get_process_num(len(total_segments))
    process_num = 24
    print(f'number of processes: {process_num}')
    processes = []

    ''' Model selection '''
    if model_name == 'BPNet':
        sig_len, samp_rate = 750, 60
    else:
        sig_len, samp_rate = 3000, 300

    ''' Multi-processing '''
    segments_per_process = np.array_split(total_segments, process_num)
    print(f'number of segments per process: {len(segments_per_process[0])}')

    for i in range(process_num):
        # list_chunk = segment_list[i * process_per_processor:(i + 1) * process_per_processor]
        list_chunk = segments_per_process[i]
        # if len(list_chunk) % 2 != 0:
        #     list_chunk = list_chunk[:-1]
        proc = mp.Process(target=read_total_data, args=(list_chunk, ple_total, abp_total, sig_len, samp_rate))
        # test = proc.pid

        processes.append(proc)
        proc.start()

    for proc in processes:
        proc.join()

    print('--- %s seconds ---' % (time.time() - start_time))
    ple_total = np.array(ple_total)
    abp_total = np.array(abp_total)
    dset = h5py.File(dset_path + str(dataset) + '.hdf5', 'w')
    dset['ple'] = ple_total
    dset['abp'] = abp_total
    dset.close()
    manager.shutdown()
    '''
    여기서 hdf5 파일로 저장
    '''

    print('total length: ', len(ple_total))
    print(np.shape(ple_total))
    print(np.shape(ple_total[0]))
    print(ple_total[0][:100])

    '''
    insert code to save ple and abp to h5py file
    '''


def dataset_split(model_name: str, data_path: str, train_ratio: float = 0.8, val_ratio: float = 0.1,
                  test_ratio: float = 0.1):
    total_segments = get_total_segment_path(data_path)
    train_segments = total_segments[:int(len(total_segments) * train_ratio)]
    val_segments = total_segments[
                   int(len(total_segments) * train_ratio):int(len(total_segments) * (train_ratio + val_ratio))]
    test_segments = total_segments[int(len(total_segments) * (train_ratio + val_ratio)):]

    multi_processing(model_name, 'train', train_segments)
    multi_processing(model_name, 'val', val_segments)
    multi_processing(model_name, 'test', test_segments)


dataset_split('BPNet', '/hdd/hdd1/dataset/bpnet/adults/physionet.org/files/mimic3wdb/1.0')

# multi_processing('BPNet', dataset='train', data_path='/hdd/hdd0/dataset/bpnet/adults/physionet.org/files/mimic3wdb/1.0')

# def dataset_handler():


