import h5py
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
import vid2bp.preprocessing.utils.signal_utils as su
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import datetime as dt

x = dt.datetime.now()
date = str(x.year) + str(x.month) + str(x.day)
ple_scale = False

if ple_scale:
    dset_path = '/hdd/hdd1/dataset/bpnet/preprocessed_' + date + '_normalized/'
    ssd_path = '/home/paperc/PycharmProjects/dataset/BPNet_mimiciii/additional2_' + date + '_normalized/'
else:
    dset_path = '/hdd/hdd1/dataset/bpnet/preprocessed_' + date + '/'
    ssd_path = '/home/paperc/PycharmProjects/dataset/BPNet_mimiciii/additional2_' + date + '/'


def add_preprocess(dataset_path: str, save_path: str, g_str: str):
    mode_list = ['train', 'val', 'test']

    for m in mode_list:
        file_path = dataset_path + m + '_' + g_str + '_0.8.hdf5'
        final_path = save_path + m + '_' + g_str + '_09.hdf5'
        survived_cnt = 0
        original_subjects = set()
        subjects = set()
        ple_total, abp_total, size_total, info_total, ohe_total = [], [], [], [], []

        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        with h5py.File(file_path, 'r') as dataset:
            ple, abp, size, info, ohe = np.array(dataset['ple']), \
                np.array(dataset['abp']), np.array(dataset['size']), np.array(dataset['info']), np.array(dataset['ohe'])
            print('total len : ', len(abp))
            for i in tqdm(range(len(ple))):
                ple_info = su.BPInfoExtractor(ple[i][0])
                abp_info = su.BPInfoExtractor(abp[i])
                subject_id = info[i].astype(int)[0]
                original_subjects.add(subject_id)
                ple_sbp = ple_info.sbp
                ple_dbp = ple_info.dbp
                abp_sbp = abp_info.sbp
                abp_dbp = abp_info.dbp
                # print('test')
                if ple_sbp[0] and ple_dbp[0] and abp_sbp[0] and abp_dbp[0] and \
                        abs(len(list(ple_sbp[-1][0])) - len(list(abp_sbp[-1][0]))) <= 2 and \
                        abs(len(ple_dbp[-1]) - len(abp_dbp[-1])) <= 2 and \
                        abs(len(list(ple_sbp[-1][0])) - len(list(ple_dbp[-1]))) <= 2 and \
                        abs(len(list(abp_sbp[-1][0])) - len(list(abp_dbp[-1]))) <= 2 and \
                        np.std(abp_sbp[-1][-1]) < 10 and np.std(ple_sbp[-1][-1]) < 0.5 and \
                        np.std(abp_dbp[-1]) < 10 and size[i][0] > 50:
                    ple_total.append(ple[i])
                    abp_total.append(gaussian_filter1d(abp[i], sigma=2))
                    dbp, sbp = np.mean(abp_dbp[-1]), np.mean(abp_sbp[-1][-1])
                    abp_sbp_max = np.max(abp[i])
                    abp_dbp_min = np.min(abp[i])
                    if sbp > abp_sbp_max:
                        sbp = abp_sbp_max
                    if dbp < abp_dbp_min:
                        dbp = abp_dbp_min
                    size_total.append([dbp, sbp, (2 * dbp + sbp) / 3])
                    info_total.append(info[i])
                    ohe_total.append(ohe[i])
                    subjects.add(subject_id)
                    survived_cnt += 1
        dset = h5py.File(final_path, 'w')
        dset['ple'] = np.array(ple_total)
        dset['abp'] = np.array(abp_total)
        dset['size'] = np.array(size_total)
        dset['info'] = np.array(info_total)
        dset['ohe'] = np.array(ohe_total)
        print('survived cnt : ', survived_cnt)
        print('ratio : ', survived_cnt / len(abp))
        print('original subjects : ', len(list(original_subjects)))
        print('subjects : ', len(list(subjects)))
        print('subject ratio : ', len(list(subjects)) / len(list(original_subjects)))


add_preprocess(dset_path, ssd_path, gender=0)
