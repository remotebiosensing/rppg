from IPython.display import display
import os
import wfdb
import numpy as np

'''
find_person(root_path)

< input >
    root_path : folder path containing person
< output > 
    ABPBVP_list : list of people that have both channel; BVP, ABP
'''


def find_person(root_path):
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
    print('channel :', len(channel))
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


def read_record(path, sampfrom=0, sampto=None):
    channel, flag = find_idx(path)
    if flag == 2:
        record = wfdb.rdrecord(path, channels=channel, sampfrom=sampfrom, sampto=sampto)
        wfdb.plot_wfdb(record=record, title='Record 039 from PhysioNet MIMIC Dataset')
        display(record.__dict__)
        return record
    else:
        print('missing signal')
        return None


'''
ple_slice(ple_sig)
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


def sig_slice(signals, size):
    print('original shape: ', np.shape(signals))
    r, c = np.shape(signals)
    print('r , c :', r, c)
    signals = np.split(signals, 2, axis=1)
    abp, ple = signals[0], signals[1]
    abp = np.reshape(abp, (size, int(r / size)))
    ple = np.reshape(ple, (size, int(r / size)))
    return abp, ple


''' 해야할 일 

'''
