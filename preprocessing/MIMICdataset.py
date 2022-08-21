# from IPython.display import display
import os
import wfdb
import numpy as np
from tqdm import tqdm
import multiprocessing

'''
find_available_data(root_path)

< input >
    root_path : folder path containing person
< output > 
    ABPBVP_list : list of people that have both channel; BVP, ABP
'''


def find_available_data(root_path):
    print('***MIMICdataset.find_available_data() called***')
    person_list = []
    for (path, dirs, files) in os.walk(root_path):
        for dir in dirs:
            if len(dir) == 3:
                person_list.append(dir)

    ABPBVP_list = []
    for person in person_list:
        channel_cnt = 0
        p = root_path + person
        for (path, dirs, files) in os.walk(p):
            for file in files:
                if file.split('.')[-1] == 'abp' or file.split('.')[-1] == 'ple':
                    channel_cnt += 1
                    if channel_cnt == 2:
                        ABPBVP_list.append(person)
                        continue
    ABPBVP_list = sorted(ABPBVP_list)

    return ABPBVP_list


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
    print('***MIMICdataset.signal_slicing() called***')
    print('in signal_slicing np.shape(signals) \n', np.shape(signals))
    nan_list1 = []
    sig = []
    drop_cnt1 = 0
    r, c = np.shape(signals)
    print('signal r, c :', r, c)
    signal_nan = np.reshape(signals, (chunk_num, int(r / chunk_num), 2))
    for s in np.isnan(signal_nan):
        nan_list1.append(s.any())
    for n, idx in enumerate(nan_list1):
        if not idx:  # pleth가 0이 아닐때 append()
            if signal_nan[n][0][1] != 0.0:
                sig.append(signal_nan[n])
        else:
            drop_cnt1 += 1

    sig = np.split(np.array(sig), 2, axis=2)
    abp, ple = np.squeeze(sig[0]), np.squeeze(sig[1])

    print('preprocessed1 : len(abp) :', len(abp), 'len(ple) :', len(ple))
    print('np.shape(abp) , np.shape(ple) :', np.shape(abp), np.shape(ple))

    return ple, abp


def data_aggregator(root_path, slicefrom=0, sliceto=None):
    print('***MIMICdataset.data_aggregator called***')
    valid_list = find_available_data(root_path)
    print('Number of total ICU patient :', len(valid_list))
    print(valid_list)
    total_data = np.empty((75000, 2))
    using_list = valid_list[slicefrom:sliceto]
    print('Number of selected ICU patient :', len(using_list))
    print(using_list)
    for l in using_list:
        ll = root_path + l
        used_file_cnt = 0
        for (path, dirs, files) in os.walk(ll):
            for file in tqdm(files):
                if len(file.split('/')[-1]) == 12 and file.split('.')[-1] == 'hea':
                    data_path = (ll + '/' + file).strip('.hea')
                    temp_data = read_record(data_path)
                    if np.shape(temp_data) == (75000, 2):
                        used_file_cnt += 1
                        total_data = np.append(total_data, temp_data, axis=0)
                        # print('np.shape(total_data) :', np.shape(total_data))
                    else:
                        print('\ntrain file dropped :', used_file_cnt, 'th ->> due to file shape :',
                              np.shape(temp_data))

    data = total_data[75000:]
    print('type of total data :', type(total_data))
    print('np.shape(total_data) :', np.shape(total_data), '== (', int(len(total_data) / 75000), '* 75000, 2 )')
    print('train file number :', used_file_cnt)
    return data
