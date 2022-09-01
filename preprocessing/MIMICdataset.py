from IPython.display import display
import os
import wfdb
import numpy as np
from tqdm import tqdm
from preprocessing.utils import math_module
from scipy import signal
import json

import h5py

with open('/home/paperc/PycharmProjects/BPNET/parameter.json') as f:
    json_data = json.load(f)
    param = json_data.get("parameters")
    orders = json_data.get("parameters").get("in_channels")
    sampling_rate = json_data.get("parameters").get("sampling_rate")

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
    channel, flag = find_idx(path)
    if flag == 2:
        record = wfdb.rdrecord(path, channels=channel, sampfrom=sampfrom, sampto=sampto)
        # wfdb.plot_wfdb(record=record, title='Record from PhysioNet MIMIC Dataset')
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


def signal_slicing(signals, chunk_num):
    print('\n***MIMICdataset.signal_slicing() called***')
    nan_check_list = []
    refined_sig = []
    dropped_nan = []  # dropped signal chunk index list ( nan included )
    dropped_zero = []  # dropped signal chunk index list ( 0.0 included )
    print(np.shape(signals))
    r, c = np.shape(signals)
    print('r :', r)
    print('c :', c)
    print('before signal_slicing >> np.shape(signals) :', np.shape(signals), '== (', int(r / 75000), '* 75000, 2 )')

    # np.shape(signal_nan) : (used_file_cnt(85) * 10, 7500, 2)
    signal_nan = np.reshape(signals, (chunk_num, int(r / chunk_num), 2))

    for s in np.isnan(signal_nan):  # np.isnan(signal_nan) : boolean list
        nan_check_list.append(s.any())  # s.any() : returns true if np.isnan(signal_nan) has "any" true

    for idx, nan in enumerate(nan_check_list):
        if not nan:
            # TODO SIGNAL_NAN에서 이상치를 찾아내는 방식을 변경해야 함 (정규분포 등등 이용)
            if (signal_nan[idx][0][1] != 0.0) and (signal_nan[idx][0][0] != 80.0) and (signal_nan[idx][0][
                                                                                           0] < 160):  # 850개 중 signal_nan의 첫번째줄(abp, ple) 값 중에 ple가 0인 값 제거 & abp == 80.0인 값 제거
                # down sampling
                refined_sig.append(signal.resample(signal_nan[idx], int(param["chunk_size"] / 125) * sampling_rate["60"]))
            else:
                dropped_zero.append(idx)
        else:
            dropped_nan.append(idx)

    dropped = sorted(dropped_zero + dropped_nan)
    print('before channel splitting :', np.shape(refined_sig))
    refined_sig = np.split(np.array(refined_sig), 2, axis=2)
    abp, ple = np.squeeze(refined_sig[0]), np.squeeze(refined_sig[1])
    print(' >> dropped_nan num :', len(dropped_nan))
    print(' >> dropped_zero num:', len(dropped_zero))
    print(' >> dropped num:', len(dropped))

    print('after signal_slicing >> np.shape(ple) :', np.shape(ple), 'np.shape(abp) :', np.shape(abp))

    # np.shape(ple) : ndarray(702, 7500), np.shape(abp) : ndarray(702, 7500)
    return ple, abp


def data_aggregator(root_path, degree=0, train=True, percent=0.7):
    print('\n***MIMICdataset.data_aggregator called*** >> degree :', degree)
    available_list = find_available_data(root_path)
    total_len = len(available_list)
    train_ = int(total_len*0.7)
    train_len = int(train_ * percent)
    print('Number of total ICU patient :', len(available_list))
    print(' >>', available_list)
    total_data = np.empty((1, 2))
    if train:
        used_list = available_list[:train_len]
    else:
        used_list = available_list[train:]
    total_cnt = 0
    print('Number of selected ICU patient :', len(used_list))
    print(' >>', used_list)
    used_file_cnt = 0
    for u in used_list:
        p = root_path + u
        for (path, dirs, files) in os.walk(p):
            for file in tqdm(files):
                if (len(file.split('/')[-1]) == 12) and (file.split('.')[-1] == 'hea'):
                    data_path = (p + '/' + file).strip('.hea')
                    temp_data = read_record(data_path)
                    total_cnt += 1
                    if np.shape(temp_data) == (75000, 2):
                        used_file_cnt += 1
                        total_data = np.append(total_data, temp_data, axis=0)
                    else:
                        print('\ntrain file dropped :', used_file_cnt, 'th ->> due to file shape is not right :',
                              np.shape(temp_data))

    data = total_data[1:]
    print('np.shape(data) :', np.shape(data))
    print('used_file_cnt :', used_file_cnt, ' / total_cnt :', total_cnt, ' ( dropped_file_cnt :',
          total_cnt - used_file_cnt, ')')

    chunk_num = int(np.shape(data)[0] / param["chunk_size"])  # / param["down_sample"])
    print('total_data_lenth :', np.shape(data)[0], ', one chunk length', param["chunk_size"])
    print('chunk_num :', chunk_num)
    print('np.shape(signal_slicing input(data) ) :', np.shape(data), '== (', int(np.shape(data)[0] / 75000),
          '* 75000, 2 )')

    if degree == 0:
        ple, abp = signal_slicing(data, chunk_num)  # f

        print('*** f data aggregation done***')
        return ple, abp, train_len
    elif degree == 1:
        ple, abp = signal_slicing(data, chunk_num)  # f
        ple_first = math_module.diff_np(ple)  # f'

        ple_total = ple_first
        abp_total = abp

        print('*** f\' data aggregation done***')
        return ple_total, abp_total, train_len
    elif degree == 2:
        ple, abp = signal_slicing(data, chunk_num)  # f
        ple_first = math_module.diff_np(ple)  # f'
        ple_second = math_module.diff_np(ple_first)  # f''

        ple_total = ple_second
        abp_total = abp
        print('*** f\'\' data aggregation done***')

        return ple_total, abp_total, train_len
    elif degree == 3:
        ple, abp = signal_slicing(data, chunk_num)  # f
        ple_first = math_module.diff_np(ple)  # f'

        ple_total = math_module.diff_channels_aggregator(ple, ple_first)
        abp_total = abp

        print('*** f & f\' data aggregation done***')
        return ple_total, abp_total, train_len
    elif degree == 4:
        ple, abp = signal_slicing(data, chunk_num)  # f
        ple_first = math_module.diff_np(ple)  # f'
        ple_second = math_module.diff_np(ple_first)  # f''

        ple_total = math_module.diff_channels_aggregator(ple, ple_second)
        abp_total = abp
        print('*** f & f\'\' data aggregation done***')

        return ple_total, abp_total, train_len
    elif degree == 5:
        ple, abp = signal_slicing(data, chunk_num)  # f
        ple_first = math_module.diff_np(ple)  # f'
        ple_second = math_module.diff_np(ple_first)  # f''

        ple_total = math_module.diff_channels_aggregator(ple_first, ple_second)
        abp_total = abp
        print('*** f\' & f\'\' data aggregation done***')

        return ple_total, abp_total, train_len
    elif degree == 6:
        ple, abp = signal_slicing(data, chunk_num)  # f
        ple_first = math_module.diff_np(ple)  # f'
        ple_second = math_module.diff_np(ple_first)  # f''

        ple_total = math_module.diff_channels_aggregator(ple, ple_first, ple_second)
        abp_total = abp
        print('*** f & f\' & f\'\' data aggregation done***')

        return ple_total, abp_total, train_len
    else:
        print('derivative not supported... goto data_aggregator()')


if __name__ == '__main__':
    root_path = param["root_path"]
    order = orders["third"]
    train_ple, train_abp, data_len = data_aggregator(root_path=root_path, degree=order[1], train=True, percent=0.075)  # 0.05 > 2 patients
    dset = h5py.File("/home/paperc/PycharmProjects/BPNET/dataset/mimic_BPNet/" + "case(" + str(order[-1])+")_len("+str(data_len)+").hdf5", "w")
    if len(train_ple) == len(train_abp):
        dset['ple'] = train_ple
        dset['abp'] = train_abp
    dset.close()

