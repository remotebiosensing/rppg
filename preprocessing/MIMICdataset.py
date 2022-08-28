from IPython.display import display
import os
import wfdb
import numpy as np
from tqdm import tqdm
from preprocessing.utils import math_module
from scipy import signal
import multiprocessing

'''
find_available_data(root_path)

< input >
    root_path : folder path containing person
< output > 
    ABPBVP_list : list of people that have both channel; BVP, ABP
'''


def find_available_data(root_path):
    print('\n***MIMICdataset.find_available_data() called***')
    person_list = []
    for (path, dirs, files) in os.walk(root_path):
        for dir in dirs:
            if len(dir) == 3:
                person_list.append(dir)

    available_list = []
    for person in person_list:
        channel_cnt = 0
        p = root_path + person
        for (path, dirs, files) in os.walk(p):
            for file in files:
                if file.split('.')[-1] == 'abp' or file.split('.')[-1] == 'ple':
                    channel_cnt += 1
                    if channel_cnt == 2:
                        available_list.append(person)
                        continue
    available_list = sorted(available_list)

    return available_list


'''
def find_idx(path)

< input >
    path : data path
< output >
    channel : index list of ABP, BVP
    flag : 2=[ABP, BVP], 1 = missing either of two
'''


def find_idx(path):
    record = wfdb.rdrecord(path)
    channel = [i for i in range(len(record.sig_name)) if (record.sig_name[i] == 'ABP' or record.sig_name[i] == 'PLETH')]
    # print('channel :', len(channel))
    flag = len(channel)
    return channel, flag


'''
def read_record(path, sampfrom=0, sampto=None)

< input >
    path : data path
    sampfrom : starting point of slice
    sampto : end point of slice
< output >    
    record : record of [ABP, BVP] channel 

wfdb.io.rdrecord()
Read a WFDB record and return the signal and record descriptors 
as attributes in a Record or MultiRecord object.
'''

''' 바로 return.p_signal 하도록 변경 '''


def read_record(path, sampfrom=0, sampto=None):
    channel, flag = find_idx(path)
    if flag == 2:
        record = wfdb.rdrecord(path, channels=channel, sampfrom=sampfrom, sampto=sampto)
        # wfdb.plot_wfdb(record=record, title='Record 039 from PhysioNet MIMIC Dataset')
        # display(record.__dict__)
        return record.p_signal
    else:
        print('read_recored() -> missing signal')
        return None


'''
signal_slicing(signals, chunk_size)
< input >
    ple_sig : plethysmo signal 
              type : np.array(75000, 2)
< output >
    slices : list of ple slice
              type : np.array(7500, 10, 2)
    
todo
1. slice a signal(75000 figure, 10 minute) into signals ( 7500 figure, 1 minute )
2. remove signal that has None values sequentially
'''


def nan_checker(signal):
    if (np.isnan(signal.any()) == False) and (np.sum(signal[0][1]) != 0):
        # print(signal)
        return signal
    else:
        print('nan list found with nan_checker')
        print(signal)
        print(signal[0][0], signal[0][1])
        return None



def signal_slicing(signals, chunk_num):
    print('\n***MIMICdataset.signal_slicing() called***')
    nan_check_list = []
    refined_sig = []
    dropped_nan = []  # dropped signal chunk index list ( nan included )
    dropped_zero = []  # dropped signal chunk index list ( 0.0 included )
    r, c = np.shape(signals)
    print('before signal_slicing >> np.shape(signals) :', np.shape(signals), '== (', int(r / 75000), '* 75000, 2 )')

    # np.shape(signal_nan) : (used_file_cnt(85) * 10, 7500, 2)
    signal_nan = np.reshape(signals, (chunk_num, int(r / chunk_num), 2))

    for s in np.isnan(signal_nan):  # np.isnan(signal_nan) : boolean list
        nan_check_list.append(s.any())  # s.any() : returns true if np.isnan(signal_nan) has "any" true

    for idx, nan in enumerate(nan_check_list):
        if not nan:
            if signal_nan[idx][0][1] != 0.0:  # 850개 중 signal_nan의 첫번째줄(abp, ple) 값 중에 ple가 0인 값 제거
                refined_sig.append(signal_nan[idx])
            else:
                dropped_zero.append(idx)
        else:
            dropped_nan.append(idx)

    dropped = sorted(dropped_zero + dropped_nan)

    refined_sig = np.split(np.array(refined_sig), 2, axis=2)
    abp, ple = np.squeeze(refined_sig[0]), np.squeeze(refined_sig[1])
    print(' >> dropped_nan num :', len(dropped_nan))
    print(' >> dropped_zero num:', len(dropped_zero))
    print(' >> dropped num:', len(dropped))
    # print(' >> dropped :', dropped)

    print('after signal_slicing >> np.shape(ple) :', np.shape(ple), 'np.shape(abp) :', np.shape(abp))

    # np.shape(ple) : ndarray(702, 7500), np.shape(abp) : ndarray(702, 7500)
    return ple, abp


def data_aggregator(root_path, degree=0, slicefrom=0, sliceto=None):
    print('\n***MIMICdataset.data_aggregator called*** >> degree :', degree)
    available_list = find_available_data(root_path)
    print('Number of total ICU patient :', len(available_list))
    print(' >>', available_list)
    total_data = np.empty((1, 2))
    used_list = available_list[slicefrom:sliceto]
    total_cnt = 0
    print('Number of selected ICU patient :', len(used_list))
    print(' >>', used_list)

    down_sample = 1800

    for u in used_list:
        p = root_path + u
        used_file_cnt = 0
        for (path, dirs, files) in os.walk(p):
            for file in tqdm(files):
                if len(file.split('/')[-1]) == 12 and file.split('.')[-1] == 'hea':
                    data_path = (p + '/' + file).strip('.hea')
                    temp_data = read_record(data_path)
                    total_cnt += 1
                    if np.shape(temp_data) == (75000, 2):
                        used_file_cnt += 1
                        ## down_Sampling
                        temp_data = signal.resample(temp_data, down_sample * 10)
                        total_data = np.append(total_data, temp_data, axis=0)
                    else:
                        print('\ntrain file dropped :', used_file_cnt, 'th ->> due to file shape is not right :',
                              np.shape(temp_data))

    data = total_data[1:]
    print('np.shape(total_data) :', np.shape(data))
    print('total_cnt :', total_cnt, '/ used_file_cnt :', used_file_cnt, ' ( dropped_file_cnt :',
          total_cnt - used_file_cnt, ')')
    chunk_num = int(np.shape(data)[0] / down_sample)
    print('np.shape(data_aggregator input) :', np.shape(data), '== (', int(len(data) / down_sample*10), '* 75000, 2 )')

    if degree == 0:
        ple, abp = signal_slicing(data, chunk_num)  # f

        print('*** f data aggregation done***')
        return ple, abp
    elif degree == 1:
        ple, abp = signal_slicing(data, chunk_num)  # f
        ple_first, abp_first = math_module.diff_np(ple, abp)  # f'

        ple_total = math_module.diff_channels_aggregator(ple, ple_first)
        abp_total = abp

        print('*** f & f\' data aggregation done***')
        return ple_total, abp_total
    elif degree == 2:
        ple, abp = signal_slicing(data, chunk_num)  # f
        ple_first, abp_first = math_module.diff_np(ple, abp)  # f'
        ple_second, abp_second = math_module.diff_np(ple_first, abp_first)  # f''

        ple_total = math_module.diff_channels_aggregator(ple, ple_first, ple_second)
        abp_total = abp
        print('*** f & f\' & f\'\' data aggregation done***')

        return ple_total, abp_total
    else:
        print('derivative not supported... goto data_aggregator()')
