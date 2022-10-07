from IPython.display import display
import os
import wfdb
import numpy as np
from tqdm import tqdm
import vid2bp.preprocessing.utils.math_module as mm
import vid2bp.preprocessing.utils.signal_utils as su
from scipy import signal
import json

import h5py

# TODO DRAW GRAPH OF PPG, VPG, APG Done
with open('/home/paperc/PycharmProjects/VBPNet/config/parameter.json') as f:
    json_data = json.load(f)
    param = json_data.get("parameters")
#     orders = json_data.get("parameters").get("in_channels")
#     sampling_rate = json_data.get("parameters").get("sampling_rate")

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
    channel, flag = find_idx(path)
    if flag == 2:
        record = wfdb.rdrecord(path, channels=channel, sampfrom=sampfrom, sampto=sampto)
        # wfdb.plot_wfdb(record=record, title='Record from PhysioNet MIMIC Dataset')
        # display(record.__dict__)
        return record.p_signal
    else:
        print('read_recored() -> missing signal')
        return None





def data_aggregator(root_path, degree=0, train=True, percent=0.7):
    print('\n***MIMICdataset.data_aggregator called*** >> degree :', degree)
    available_list = find_available_data(root_path)
    total_len = len(available_list)
    train_ = int(total_len * 0.7)
    train_len = int(train_ * percent)
    print('Number of total ICU patient :', len(available_list))
    print(' >>', available_list)
    total_data = np.empty((1, 2))
    if train:
        used_list = available_list[:train_len]
    else:
        used_list = available_list[train_len:]
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
    print('total_data_lenth :', np.shape(data)[0], ', one chunk length', param["chunk_size"])
    print('np.shape(signal_slicing input(data) ) :', np.shape(data), '== (', int(np.shape(data)[0] / 75000),
          '* 75000, 2 )')

    if degree == 0:
        abp, ple = su.signal_slicing(data)  # f
        print('*** f data aggregation done***')

        return abp, ple, train_len
    elif degree == 1:
        abp, ple = su.signal_slicing(data)  # f
        ple_first = mm.diff_np(ple)  # f'
        ple_total = ple_first
        print('*** f\' data aggregation done***')

        return abp, ple_total, train_len
    elif degree == 2:
        abp, ple = su.signal_slicing(data)  # f
        ple_first = mm.diff_np(ple)  # f'
        ple_second = mm.diff_np(ple_first)  # f''
        ple_total = ple_second
        print('*** f\'\' data aggregation done***')

        return abp, ple_total, train_len

    elif degree == 3:
        abp, ple = su.signal_slicing(data)  # f
        # ple = su.scaler(ple)
        ple_first = mm.diff_np(ple)  # f'
        ple_total = mm.diff_channels_aggregator(ple, ple_first)
        print('*** f & f\' data aggregation done***')

        return abp, ple_total, train_len

    elif degree == 4:
        abp, ple = su.signal_slicing(data)  # f
        ple_first = mm.diff_np(ple)  # f'
        ple_second = mm.diff_np(ple_first)  # f''
        ple_total = mm.diff_channels_aggregator(ple, ple_second)
        print('*** f & f\'\' data aggregation done***')

        return abp, ple_total, train_len

    elif degree == 5:
        abp, ple = su.signal_slicing(data)  # f
        ple_first = mm.diff_np(ple)  # f'
        ple_second = mm.diff_np(ple_first)  # f''
        ple_total = mm.diff_channels_aggregator(ple_first, ple_second)
        print('*** f\' & f\'\' data aggregation done***')

        return abp, ple_total, train_len

    elif degree == 6:
        abp, ple = su.signal_slicing(data)  # f
        ple_first = mm.diff_np(ple)  # f'
        ple_second = mm.diff_np(ple_first)  # f''
        ple_total = mm.diff_channels_aggregator(ple, ple_first, ple_second)
        print('*** f & f\' & f\'\' data aggregation done***')

        return abp, ple_total, train_len

    else:
        print('derivative not supported... goto data_aggregator_sig_processed()')



if __name__ == '__main__':
    root_path = param["root_path"]
    order = orders["third"]
    train_ple, train_abp, data_len = data_aggregator(root_path=root_path, degree=order[1], train=True, percent=0.075)  # 0.05 > 2 patients
    dset = h5py.File("/home/paperc/PycharmProjects/BPNET/dataset/mimic_BPNet/" + "case(" + str(order[-1])+")_len("+str(data_len)+").hdf5", "w")
    if len(train_ple) == len(train_abp):
        dset['ple'] = train_ple
        dset['abp'] = train_abp
    dset.close()

