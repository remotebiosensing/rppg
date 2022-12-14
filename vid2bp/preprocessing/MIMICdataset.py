from IPython.display import display
import os
import wfdb
import numpy as np
from tqdm import tqdm
import vid2bp.preprocessing.utils.signal_utils as su

'''
find_available_data(root_path)

< input >
    root_path : folder path containing person
< output > 
    ABPBVP_list : list of people that have both channel; BVP, ABP
'''


def find_available_data(root_path):
    print('\n***MIMICdataset.find_available_data() called***')
    print(root_path)
    person_list = []
    for (path, dirs, files) in os.walk(root_path):
        for d in dirs:
            if len(d) == 3:
                person_list.append(d)

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


# TODO -> Record.inti_value를 추가로 return 해주는데, 뒤에 함수들 전부 수정
def read_record(path, sampfrom=0, sampto=None):
    # channel = ['ABP', 'PLETH']
    channel, flag = find_idx(path)
    if flag == 2:
        record = wfdb.rdrecord(path, channels=channel, sampfrom=sampfrom, sampto=sampto)
        wfdb.plot_wfdb(record=record, title='Record from PhysioNet MIMIC Dataset')
        display(record.__dict__)
        return record.p_signal
    else:
        print('read_recored() -> missing signal')
        return None


def data_aggregator(model_name, read_path, chunk_size, samp_rate):
    print('MIMIC-I dataset selected')
    available_list = find_available_data(read_path)
    print('Number of total ICU patient :', len(available_list))
    print(' >>', available_list)
    total_data = np.empty((1, 2))
    total_cnt = 0
    print('Number of selected ICU patient :', len(available_list))
    used_file_cnt = 0
    # TODO ''' use total data after solving the problem of get_diastolic() '''
    for a in available_list[:1]:
        p = read_path + a
        for (path, dirs, files) in os.walk(p):
            for file in tqdm(files):
                if (len(file.split('/')[-1]) == 12) and (file.split('.')[-1] == 'hea'):
                    data_path = (p + '/' + file).strip('.hea')
                    used_file_cnt += 1
                    total_data = np.append(total_data, read_record(data_path), axis=0)

    sig_total = total_data[1:]
    # for reshaping in signal_utils
    relength = (len(sig_total) // chunk_size) * chunk_size
    sig_total = sig_total[:relength]
    print('np.shape(sig_total) :', np.shape(sig_total))

    abp, ple, dsm = su.signal_slicing(model_name, sig_total, chunk_size, samp_rate, fft=True)
    return ple, abp, dsm
