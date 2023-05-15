import os
import random
import warnings
import datetime as dt

import json
import wfdb
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import signal

from cnibp.preprocessing.utils.signal_utils import signal_comparator
from cnibp.preprocessing.utils.signal_utils import DualSignalHandler as dsh
from cnibp.preprocessing.utils import mp_functions as mf, wfdb_functions as wf
import cnibp.preprocessing.mimiciii_matched as match_prep


warnings.filterwarnings("ignore", message='Degrees of freedom <= 0 for slice')
warnings.filterwarnings("ignore", message='Mean of empty slice')


'''
Dataset Structure

grouped_by_patient_id
                    - patients
                            - segments
                                    - chunks

'''

with open('/home/paperc/PycharmProjects/Pytorch_rppgs/cnibp/configs/parameter.json') as f:
    json_data = json.load(f)
    # param = json_data.get("parameters")
    channels = json_data.get("parameters").get("in_channels")
    gender_info = json_data.get("parameters").get("gender")


def list_shuffler(path_list):
    """
    shuffles patient's segments
    :param path_list: list of path to be shuffled
    :return: shuffled path_list
    """
    shuffle_cnt = random.randint(len(path_list) * 2, len(path_list) * 3)
    for _ in range(shuffle_cnt):
        i = random.randint(0, len(path_list) - 1)
        j = random.randint(0, len(path_list) - 1)
        path_list[i], path_list[j] = path_list[j], path_list[i]
    random.shuffle(path_list)
    return path_list


def sort_by_fsize(segments: list, size_list: list):
    split_by_size = []
    sorted_by_fsize = sorted(segments, key=lambda fs: os.stat(fs.replace('.hea', '.dat')).st_size)
    if 'light0' in size_list:
        split_by_size.append(sorted_by_fsize[:int(len(sorted_by_fsize) * 0.25)])
    if 'light1' in size_list:
        split_by_size.append(sorted_by_fsize[int(len(sorted_by_fsize) * 0.25):int(len(sorted_by_fsize) * 0.4)])
    if 'light2' in size_list:
        split_by_size.append(sorted_by_fsize[int(len(sorted_by_fsize) * 0.4):int(len(sorted_by_fsize) * 0.55)])
    if 'light3' in size_list:
        split_by_size.append(sorted_by_fsize[int(len(sorted_by_fsize) * 0.55):int(len(sorted_by_fsize) * 0.70)])
    if 'heavy1' in size_list:
        split_by_size.append(sorted_by_fsize[int(len(sorted_by_fsize) * 0.70):int(len(sorted_by_fsize) * 0.85)])
    if 'heavy2' in size_list:
        split_by_size.append(sorted_by_fsize[int(len(sorted_by_fsize) * 0.80):int(len(sorted_by_fsize) * 0.95)])

    return split_by_size


def get_segments_per_person(patient_path: str):
    """
    :param patient_path: path of a single patient (e.g. /hdd/hdd0/dataset/bpnet/adults/physionet.org/files/mimic3wdb/1.0/30/3001937_11)
    :return: all_file_paths: list of all segments of a single patient
    """
    segment_path = []
    for root, dirs, files in os.walk(patient_path):
        for file in files:
            if file.endswith('.hea'):
                segment_path.append(root + '/' + file)
    segment_path = [p for p in segment_path if
                    p.endswith('.hea') and ('_' in p.split('/')[-1]) and ('layout' not in p)]
    return segment_path


def get_segments_path(read_path: str, gen: int, encoder: str, segment_max_cnt: int, size_list: list):
    total_patient_path = []
    train_patient, val_patient, test_patient = [], [], []
    train_segments, val_segments, test_segments = [], [], []
    # encoder = 'DIAGNOSIS'
    for roots, dirs, files in os.walk(read_path):
        if len(files) > 0:
            total_patient_path.append(roots)
        else:
            continue
    total_patient_path = [p for p in total_patient_path if '_' in p.split('/')[-1]]
    pid = [int(t.split('/')[-1].split('_')[0][-5:]) for t in total_patient_path]
    # for p in pid:
    #     pp[str(p)]
    # pid_list = [[int(t.split('/')[-1].split('_')[0][-5:]), t] for t in total_patient_path]
    pid_dict = {}
    for t in total_patient_path:
        pid_dict[str(t.split('/')[-1].split('_')[0][-5:])] = t
    # pid_list = [{str(t.split('/')[-1].split('_')[0][-5:]) : t} for t in total_patient_path]

    patient_info, patient_df, le = match_prep.get_patients_info(gen, pid, encoder_type=encoder, )

    for i in range(len(le.classes_)):
        # labels = le.inverse_transform([1,2,3,4,5])
        # split = patient_df[patient_df['reDIAGNOSIS'] == le.classes_[i]]['SUBJECT_ID']
        class_i = patient_df[patient_df[encoder + '_LABEL'] == i]['SUBJECT_ID'].to_numpy()
        split = np.vsplit(np.reshape(class_i, (-1, 1)), [int(len(class_i) * 0.8), int(len(class_i) * 0.9)])
        train_patient.extend(split[0])
        val_patient.extend(split[1])
        test_patient.extend(split[2])
    train_patient = list(np.squeeze(train_patient))
    val_patient = list(np.squeeze(val_patient))
    test_patient = list(np.squeeze(test_patient))

    if len(set(train_patient) & set(test_patient)) > 0:
        print('train and test set has same patient due to different admission history, removing from test set...')
        test_patient = list(set(test_patient) - set(train_patient))

    for tr in train_patient:
        segments = get_segments_per_person(pid_dict[str(format(tr, '05d'))])
        if len(segments) > segment_max_cnt:
            train_segments.extend(list_shuffler(segments)[:segment_max_cnt])
        else:
            train_segments.extend(segments)
    for val in val_patient:
        segments = get_segments_per_person(pid_dict[str(format(val, '05d'))])
        if len(segments) > segment_max_cnt:
            val_segments.extend(list_shuffler(segments)[:segment_max_cnt])
        else:
            val_segments.extend(segments)
    for te in test_patient:
        segments = get_segments_per_person(pid_dict[str(format(te, '05d'))])
        if len(segments) > segment_max_cnt:
            test_segments.extend(list_shuffler(segments)[:segment_max_cnt])
        else:
            test_segments.extend(segments)
    print('total number of patients: ', len(total_patient_path))
    print('total number of train segments: ', len(train_segments))
    print('total number of val segments: ', len(val_segments))
    print('total number of test segments: ', len(test_segments))

    return sort_by_fsize(list_shuffler(train_segments), size_list), \
           sort_by_fsize(list_shuffler(val_segments), size_list), \
           sort_by_fsize(list_shuffler(test_segments), size_list), \
           patient_info, patient_df


def read_total_data(process_id: int, segments: list,
                    preprocessing_mode: str, chunk_size: int, sampling_rate: int,
                    threshold: float, ple_scale: bool, hdf_flag: bool,
                    patient_info_df, ple_cycle: list, abp_cycle: list,
                    info_total: list, ple_total: list, abp_total: list, dbp_total: list, sbp_total: list,
                    p_status_total: list, a_status_total: list):
    """

    * if a single record is shorter than 6 seconds, skip it to consider only long enough to have respiratory cycles
       else, slice it into 6 seconds segments

    ** if a single record is too long,
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
    total_chunk_cnt = 0
    for segment in tqdm(segments, desc='process-' + str(process_id), leave=False):
        total_chunk_cnt += 1
        chunk_per_segment = 0
        segment = segment.strip('.hea')
        patient_id = segment.split('/')[-2].split('_')[0][-5:]
        ple_idx, abp_idx = wf.find_channel_idx(segment)
        raw_ple_segment, raw_abp_segment = np.squeeze(
            np.split(wfdb.rdrecord(segment, channels=[ple_idx, abp_idx]).p_signal, 2, axis=1))
        if raw_abp_segment is float or raw_ple_segment is float or len(raw_ple_segment) < chunk_size or len(
                raw_abp_segment) < chunk_size:
            continue
        else:
            nan_mask = ~(np.isnan(raw_ple_segment) | np.isnan(raw_abp_segment))
            if ple_scale:
                digital_sig, gain, baseline = wf.get_channel_record(segment, ple_idx)
                ple_record = wfdb.rdrecord(segment, channels=[ple_idx])
                ple_digital_sig = np.squeeze(ple_record.adc())
                ple_gain = ple_record.adc_gain[0]
                ple_baseline = ple_record.baseline[0]
                ''' ************************************************** revert gain controller '''
                if gain != 1023.:
                    ple_segment = (((ple_digital_sig - ple_baseline) / ple_gain) * (1023. / ple_gain))[nan_mask]
                else:
                    ple_segment
                ''' ************************************************************************* '''
            else:
                ple_segment = raw_ple_segment[nan_mask]
            abp_segment = raw_abp_segment[nan_mask]

            if len(ple_segment) < chunk_size or len(abp_segment) < chunk_size:
                continue

            ple_chunks, abp_chunks = dsh(ple_segment, abp_segment).shuffle_lists()
            for p_chunk, a_chunk in zip(ple_chunks, abp_chunks):
                p_flat_range = np.where(p_chunk == np.max(p_chunk))[0]
                a_flat_range = np.where(a_chunk == np.max(a_chunk))[0]
                if len(p_flat_range) > len(p_chunk) * 0.3 or len(a_flat_range) > len(p_chunk) * 0.3:
                    continue
                '''-----------------------------------------'''
                confirm_flag, ple_info, abp_info = signal_comparator(p_chunk, a_chunk,
                                                                     preprocessing_mode, threshold, sampling_rate)
                if confirm_flag is hdf_flag:  # 학습에 사용할 데이터셋 # ple_info.valid_flag and abp_info.valid_flag and corr > 0.9
                    chunk_per_segment += 1
                    if chunk_per_segment == 10:
                        break
                    ple_total.append(ple_info.input_sig)
                    ple_cycle.append(signal.resample(ple_info.cycle, 100))
                    abp_total.append(abp_info.input_sig)
                    abp_cycle.append(signal.resample(abp_info.cycle, 100))
                    dbp_total.append(dsh(abp_info.dbp_idx, abp_info.dbp_value).stack_sigs('vertical', resize_n=15))
                    sbp_total.append(dsh(abp_info.sbp_idx, abp_info.sbp_value).stack_sigs('vertical', resize_n=15))
                    info_total.append(patient_info_df[patient_id][:-1])
                    p_status_total.append(ple_info.status)
                    a_status_total.append(abp_info.status + abp_info.detail_status)
                    # p_status_total.append(ple_info.status)
                    # a_status_total.append(abp_info.status)
                else:  # 분석에 사용할 데이터셋
                    continue
                    # invalid_pid.append(patient_id)
                    # invalid_p_chunk.appepnd(p_chunk)
                    # invalid_a_chunk.append(a_chunk)
                    # p_status_total.append(np.array(ple_info.status))
                    # a_status_total.append(np.array(abp_info.status + abp_info.detail_status))
                    #
                    # chunk_per_segment += 1
                    # ''' start adding data to hdf5 file '''
                # else:
                #     flipped_chunk_cnt += 1
                #     continue
                #     p_cycle = p_sig_info.get_cycle()
                #     a_cycle = a_sig_info.get_cycle()
                # ple_total.append(p_chunk)
                # abp_total.append(a_chunk)
                # size_total.append(len(p_chunk))
                # ohe_total.append(patient_info_total[patient_id])
                # eliminated_total.append(0)
            # pass


def split_dataset(model_name: str, params: dict, train_segments: list, val_segments: list, test_segments: list,
                  patient_info, patient_df, data_size: list):
    '''
    model_name : selects the length of chunks
    root_path : dataset's root path
    gender : selects gender , takes string ( 'Total', 'Male', 'Female' )
    threshold : correlation between ppg signal and abp signal
    ple_scale : if True, returns restored ppg signal with gain
    '''

    # len_info = []
    g_info = gender_info[params['gender']]
    x = dt.datetime.now()
    date = str(x.year) + str(x.month) + str(x.day)
    dset_path = '/hdd/hdd1/dataset/bpnet/preprocessed_' + date
    # ssd_path = '/home/paperc/PycharmProjects/dataset/BPNet_mimiciii/additional' + date
    if params['ple_scale']:
        dset_path += '_restored/'
        # ssd_path += '_restored/'
    dset_path += str(params['mode']) + '/'

    if not os.path.isdir(dset_path):
        os.makedirs(dset_path)
    # if not os.path.isdir(ssd_path):
    #     os.mkdir(ssd_path)

    # print('dataset splitting...')
    # train_segments, val_segments, test_segments, patient_info, patient_df = get_segments_path(root_path, g_info[0],
    #                                                                                           encoder=params['encoder'],
    #                                                                                           segment_max_cnt=params[
    #                                                                                               'segment_per_patient'],
    #                                                                                           size_list=f_list)

    patient_df.to_csv(dset_path + 'patient_data.csv')
    # patient_df.to_csv(ssd_path + 'patient_data.csv')
    # print('dataset splitting done... signal extracting...')

    for (m, s) in zip(['Train', 'Val', 'Test'], [train_segments, val_segments, test_segments]):
        print('extracting ' + m + ' data...')
        data_len_list = mf.multi_processing_sort_by_file_size(model_name=model_name, target_function=read_total_data,
                                                              mode=m,
                                                              parameters=params,
                                                              dset_path=dset_path,
                                                              splitted_segments_by_f_size=s,
                                                              patient_info_df=patient_info)
        data_size.append(data_len_list)
        print('extracting ' + m + ' data done...')

    # mf.multi_processing_sort_by_file_size(model_name=model_name, target_function=read_total_data, mode='Train',
    #                                       parameters=params,
    #                                       dset_path=dset_path,
    #                                       segments=train_segments,
    #                                       patient_info_df=patient_info)


if __name__ == "__main__":
    len_info = []

    # f_size_list = ['light0', ['light1', 'light2', 'light3', 'heavy1', 'heavy2']
    f_size_list = ['light3', 'heavy1', 'heavy2']
    preprocessing_mode = ['none', 'damp', 'flat', 'flip', 'total']

    params = {
        'sampling_rate': 125,
        'chunk_size': 750,
        'mode': 'temp',
        'hdf_flag': True,
        'ple_scale': True,
        'corr_threshold': 0.9,
        'gender': 'Total',
        'encoder': 'DIAGNOSIS',
        'segment_per_patient': 15,
        'chunk_per_segment': 20}
    root_path = '/hdd/hdd1/dataset/mimiciiisubset/physionet.org/files/mimic3wdb-matched/1.0'
    g_info = gender_info[params['gender']]

    train_segments, val_segments, test_segments, patient_info, patient_df = get_segments_path(root_path, g_info[0],
                                                                                              encoder=params['encoder'],
                                                                                              segment_max_cnt=params[
                                                                                                  'segment_per_patient'],
                                                                                              size_list=f_size_list)

    for m in preprocessing_mode:
        params['mode'] = m
        split_dataset(model_name='BPNet',
                      params=params,
                      train_segments=train_segments,
                      val_segments=val_segments,
                      test_segments=test_segments,
                      patient_info=patient_info,
                      patient_df=patient_df,
                      data_size=len_info)

        # df_col1.append(m)
    f_size_list.append('total')
    df = pd.DataFrame(np.array(len_info).transpose(), index=f_size_list,
                      columns=[np.array([[x] * 3 for x in preprocessing_mode]).flatten(),
                               ['Train', 'Validation', 'Test'] * len(preprocessing_mode)])
    df.to_csv('/home/paperc/PycharmProjects/Pytorch_rppgs/cnibp/csv/' + 'dataset_size.csv', sep=',', na_rep='NaN')
    print('preprocessing done...')
