import os
import wfdb
import random

from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Process, shared_memory, Semaphore
import numpy as np
import vid2bp.preprocessing.utils.signal_utils as su
import matplotlib.pyplot as plt
import vid2bp.preprocessing.utils.multi_processing as multi
import vid2bp.preprocessing.utils.data_shuffler as ds
import time


def get_processor_num(target_num: int):
    divisors = []
    for i in range(1, int(target_num ** 0.5) + 1):
        if target_num % i == 0:
            divisors.append(i)
            if i != target_num // i:
                divisors.append(target_num // i)
    test = [x for x in divisors if x < os.cpu_count()]
    if np.max(test) < os.cpu_count() // 2:
        return os.cpu_count()
    else:
        return np.max(test)


def get_segments_per_person(read_path: str):
    all_file_paths = []
    for root, dirs, files in os.walk(read_path):
        for file in files:
            if file.endswith('.hea'):
                all_file_paths.append(root + '/' + file)
    all_file_paths = [p for p in all_file_paths if
                      p.endswith('.hea') and ('_' in p.split('/')[-1]) and ('layout' not in p)]
    random.shuffle(all_file_paths)
    return all_file_paths


# temp_list = get_segments_per_person('/hdd/hdd0/dataset/bpnet/temp/physionet.org/files/mimic3wdb/1.0')

def get_all_shuffled_segments(read_path: str):
    '''
    param:
        read_path: path of the dataset (e.g. /hdd/hdd0/dataset/bpnet/adults/physionet.org/files/mimic3wdb/1.0)
    return:
        all_patient_paths: list of all patient paths
    '''
    all_patient_paths = []
    all_shuffled_data_path = []
    # 사람별로 가져오는 작업(all_patient_paths), 1700개 segment 있는 사람도 있고, 1개 있는 사람도 있고
    for root, dirs, files in os.walk(read_path):
        if len(files) > 0:
            all_patient_paths.append(root)
    all_patient_paths = [p for p in all_patient_paths if '_' in p.split('/')[-1]]
    # 가져온 사람 p를 돌면서 p 안에 있는 segment를 최대 35개 가져옴
    for p in all_patient_paths:
        segments_per_person = get_segments_per_person(p)
        if len(segments_per_person) > 10:
            # 한 사람의 segment가 10개가 넘으면, 랜덤하게 10개를 뽑아서 가져옴
            random.shuffle(segments_per_person)
            segments_per_person = segments_per_person[:10]
        all_shuffled_data_path.extend(segments_per_person)
    # 사람 순서대로 가져온 segments_per_person을 랜덤하게 다시 셔플
    random.shuffle(all_shuffled_data_path)
    return all_shuffled_data_path


# t_list = get_all_shuffled_segments('/hdd/hdd0/dataset/bpnet/temp/physionet.org/files/mimic3wdb/1.0')


'''
어제 파일 경로로 셔플하는 것까지는 완료했는데, 문제는 사람 단위로 overfitting 되는 것 방지하기 위해 378736_722 같은 애들은 30개로 짤라야 함



'''

# get all file paths

'''** mimic 전체로 통합'''


def find_idx(path):
    '''
    param:
        path: path of a segment (e.g. /hdd/hdd0/dataset/bpnet/adults/physionet.org/files/mimic3wdb/1.0/30/3001937_11)
    return:

    '''
    record = wfdb.rdrecord(path)
    ppg_idx = [p for p in range(len(record.sig_name)) if record.sig_name[p] == 'PLETH']
    abp_idx = [a for a in range(len(record.sig_name)) if record.sig_name[a] == 'ABP']
    return ppg_idx, abp_idx





def read_total_data(segment_list: list, ppg_total: list, abp_total: list, chunk_size: int):
    '''
    ** if single patient have too many records,
       randomly select 30(or any number you like) records to prevent over-fitting to single patient

    ** if single segment is shorter than 6 seconds, skip it
       else, slice it into 6 seconds segments

    param:
        path: path of a patient (e.g. /hdd/hdd0/dataset/bpnet/adults/physionet.org/files/mimic3wdb/1.0/30/3001937_11)
        sampfrom: start index of the segment
        sampto: end index of the segment
    return:
        record: wfdb record object containing PLETH and ABP signals
        patient_records: list of wfdb record
    '''

    for segment in tqdm(segment_list):
        ppg = wfdb.rdrecord(segment.strip('.hea'), channels=find_idx(segment.strip('.hea'))[0]).p_signal
        abp = wfdb.rdrecord(segment.strip('.hea'), channels=find_idx(segment.strip('.hea'))[1]).p_signal
        if len(ppg) > chunk_size and len(abp) > chunk_size:
            for i in range(len(ppg) // chunk_size):
                '''
                여기서 하나의 chunk마다 ppg와 abp preprocessing을 할 것인지
                '''
                ppg_total.append(np.squeeze(ppg[i * chunk_size:(i + 1) * chunk_size]))
                abp_total.append(np.squeeze(abp[i * chunk_size:(i + 1) * chunk_size]))
        else:
            continue

    '''
    아니면 여기서 전체를 다시 순회하면서 preprocessing을 할 것인지장
    multi_processing에서 h5py 파일 저
    '''

# def read_total_data_shm(id, segment_list: list, shm_name, ppg_shm, abp_shm, sem):
#     chunk_size = 750
#     ppg_temp =[]
#     abp_temp = []
#     print(id)
#     print(type(id))
#     for segment in tqdm(segment_list):
#         ppg = wfdb.rdrecord(segment.strip('.hea'), channels=find_idx(segment.strip('.hea'))[0]).p_signal
#         abp = wfdb.rdrecord(segment.strip('.hea'), channels=find_idx(segment.strip('.hea'))[1]).p_signal
#         if len(ppg) > chunk_size and len(abp) > chunk_size:
#             for i in range(len(ppg) // chunk_size):
#                 '''
#                 여기서 하나의 chunk마다 ppg와 abp preprocessing을 할 것인지
#                 '''
#                 ppg_temp.append(np.squeeze(ppg[i * chunk_size:(i + 1) * chunk_size]))
#                 abp_temp.append(np.squeeze(abp[i * chunk_size:(i + 1) * chunk_size]))
#         else:
#             continue
#     sem.acquire()
#     new_shm = shared_memory.SharedMemory(name=shm_name)
#     ppg_arr = np.ndarray(ppg_shm.shape, dtype=ppg_shm.dtype, buffer=new_shm.buf)
#     abp_arr = np.ndarray(abp_shm.shape, dtype=abp_shm.dtype, buffer=new_shm.buf)
#     ppg_arr[int(id)] = ppg_temp
#     abp_arr[int(id)] = abp_temp
#     # tmp_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=new_shm.buf)
#     sem.release()
#     print(f'{id}번째 프로세스가 끝났습니다.')
#
# def multi_processing_shm(segment_list: list, chunk_size: int = 750, num_workers: int = 24):
#     arr = np.empty((chunk_size,))
#     # arr = np.array([])
#
#     shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
#     ppg_shm = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
#     abp_shm = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
#
#     sem = Semaphore()
#
#     processes = []
#     # chunk_list에는 각 process가 처리할 segment list가 들어있음 (e.g. [[1,2,3], [4,5,6], [7,8,9]])
#     chunk_list = np.array_split(segment_list, num_workers)
#     for i in range(num_workers):
#         p = Process(target=read_total_data_shm, args=(i, chunk_list[i], shm.name, ppg_shm, abp_shm, sem))
#         processes.append(p)
#         p.start()
#
#     for p in processes:
#         p.join()
#     pass
#     # '''
#     # param:
#     #     segment_list: list of all segments
#     #     chunk_size: size of each chunk
#     #     num_workers: number of workers
#     # return:
#     #     ppg_total: list of ppg chunks
#     #     abp_total: list of abp chunks
#     # '''
#     # ppg_total = []
#     # abp_total = []
#     # # segment_list를 num_workers로 나누기
#     # segment_list = np.array_split(segment_list, num_workers)
#     # # multiprocessing을 위한 shared memory 생성
#     # shm = shared_memory.SharedMemory(create=True, size=chunk_size * 2 * 10000)
#     # arr = np.ndarray((chunk_size * 2, 10000), dtype=np.float32, buffer=shm.buf)
#     # # multiprocessing을 위한 semaphore 생성
#     # sem = multiprocessing.Semaphore()
#     # # multiprocessing을 위한 process 생성
#     # processes = []
#     # for i in range(num_workers):
#     #     p = multiprocessing.Process(target=read_total_data_shm, args=(i, segment_list[i], ppg_total, abp_total, shm, arr, sem))
#     #     processes.append(p)
#     #     p.start()
#     # for p in processes:
#     #     p.join()
#     # shm.close()
#     # shm.unlink()
#     # return ppg_total, abp_total


def multi_processing(model_name, data_path: str):
    '''
    param:
        data_path: path of the dataset (e.g. /hdd/hdd0/dataset/bpnet/adults/physionet.org/files/mimic3wdb/1.0)
        chunk_size: size of slice to be cut from the signal
    return:
        None
    '''
    # arr = np.empty((1,750))
    # for i in range(100):
    #     temp = np.ones((1,750))
    #     arr = np.append(arr,temp)
    start_time = time.time()
    segment_list = get_all_shuffled_segments(data_path)

    manager = mp.Manager()
    ppg_total = manager.list()
    abp_total = manager.list()

    total_len = len(segment_list)
    processor_num = get_processor_num(total_len)
    # process_per_processor = total_len // processor_num
    # res = total_len % processor_num
    processes = []

    ''' Model selection '''
    if model_name == 'BPNet':
        sig_len, samp_rate = 750, 60
    else:
        sig_len, samp_rate = 3000, 300

    ''' Multi-processing '''
    list_split = np.array_split(segment_list, processor_num)

    for i in range(processor_num):
        # list_chunk = segment_list[i * process_per_processor:(i + 1) * process_per_processor]
        list_chunk = list_split[i]
        proc = mp.Process(target=read_total_data, args=(list_chunk, ppg_total, abp_total, sig_len))
        processes.append(proc)
        # proc.start()
    for i in processes:
        i.start()
    for proc in processes:
        proc.join()

    # if res != 0:
    #     read_total_data(segment_list[-res:], ppg_total, abp_total, sig_len)

    ppg_total = np.array(ppg_total)
    abp_total = np.array(abp_total)
    print('--- %s seconds ---' % (time.time() - start_time))

    print('total length: ', len(ppg_total))
    print(np.shape(ppg_total))
    print(np.shape(ppg_total[0]))
    print(ppg_total[0][:100])

    manager.shutdown()
    '''
    insert code to save ppg and abp to h5py file
    '''


# adult_path = '/hdd/hdd0/dataset/bpnet/temp/physionet.org/files/mimic3wdb/1.0'
# ll =get_all_shuffled_segments('/hdd/hdd0/dataset/bpnet/temp/physionet.org/files/mimic3wdb/1.0')[:100]
# multi_processing_shm(ll)

multi_processing('BPNet', '/hdd/hdd0/dataset/bpnet/temp/physionet.org/files/mimic3wdb/1.0')
