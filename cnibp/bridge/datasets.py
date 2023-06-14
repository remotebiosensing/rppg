import csv
# import librosa
import os
from glob import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
from cnibp.preprocessing.utils.signal_utils import SignalInfoExtractor as extractor
from cnibp.preprocessing.utils.signal_utils import SignalHandler as handler
from scipy.signal import resample, detrend
from tqdm import tqdm

ubfc_path = '/hdd/hdd1/dataset/data/UBFC/'
mimic_path = '/hdd/hdd1/dataset/bpnet/preprocessed_2023612_restored/total/Train_Total_0.9.hdf5'
mimic_additional_path = '/hdd/hdd1/dataset/bpnet/preprocessed_2023612_restored/total/Train_Total_0.9_status.hdf5'


def load_preprocessed_mimic_data(preprocessed_path):
    # open hdf5 file from preprocessed_path
    with h5py.File(mimic_additional_path, 'r') as ff:
        # print(ff.keys())
        ple_cycle = np.array(ff['ple_cycle'])
        ple_cycle_len = np.array(ff['ple_cycle_len'])
        info = np.array(ff['info'])
    with h5py.File(preprocessed_path, 'r') as f:
        # print(f.keys())
        ple = np.array(f['ple'])
    offset = 150
    for o in range(5):
        for i in range(10):
            # plt.title()
            min = np.min(ple_cycle[i + offset + o])
            max = np.max(ple_cycle[i + offset])
            # plt.plot(resample((ple_cycle[i + offset + o] - min) / (max - min),
            #                   int(ple_cycle_len[i + offset + o])), label=str(ple_cycle_len[i + offset]))
        # plt.legend()
        # plt.show()
    return ple, ple_cycle, ple_cycle_len, info


def get_rppg_gt(root_path):
    if 'UBFC' in root_path:
        root_path = '/hdd/hdd1/dataset/data/UBFC/'
        # get all file path containing 'subject' from root path
        subject_paths = glob(os.path.join(root_path, '*'))
        subject_paths = sorted([s for s in subject_paths if 'subject' in s])
        # add 'ground_truth.txt' to each subject path
        subject_paths = [os.path.join(s, 'ground_truth.txt') for s in subject_paths]
        subject_num_list = [s.split('t')[-5].split('/')[0] for s in subject_paths]

        rppg_groundtruth = []
        rppg_chunks = []
        for s in subject_paths:
            with open(s, 'r') as f:
                lines = [line.rstrip() for line in f if line]
                ppg_gt = np.array(lines[0].split(), dtype=float)
                rppg_groundtruth.append(ppg_gt)
                ppg_chunks = handler(ppg_gt).list_slice(180)
                rppg_chunks.append(ppg_chunks)

    else:
        raise NotImplementedError
    # print(subject_paths)
    ple_info = extractor(rppg_groundtruth[0][:180], True, 'total')
    return subject_paths, subject_num_list, rppg_groundtruth, rppg_chunks


def get_most_similar_waveform():
    mimic_ple, mimic_ple_cycle, mimic_ple_cycle_len, mimic_info = load_preprocessed_mimic_data(mimic_path)
    _, _, rppg_gt, chunks = get_rppg_gt(ubfc_path)
    for cs in tqdm(chunks, total=len(chunks), desc='chunks', position=1):
        for c in cs:
            for m, mc, mcl in tqdm(zip(mimic_ple, mimic_ple_cycle, mimic_ple_cycle_len), total=len(mimic_ple), desc='mimic_ple', position=0):
                resampled_mimic_ple = resample(m, 180)
                resampled_mimic = extractor(resampled_mimic_ple, True, 'total')
                # corr = np.corrcoef(detrend(handler(c).normalize()), detrend(handler(resample(m, 180)).normalize()))[0][1]
                rppg_gt_cycle = handler(extractor(c, True, 'total').cycle).normalize('minmax')
                mimic_gt_cycle = handler(resampled_mimic.cycle).normalize('minmax')
                cycle_corr = np.corrcoef(rppg_gt_cycle, mimic_gt_cycle)[0][1]
                if cycle_corr > 0.8:
                    plt.title(str(cycle_corr))
                    plt.plot(handler(detrend(c)).normalize('minmax'), label='rppg')
                    plt.plot(handler(detrend(resample(m, 180))).normalize('minmax'), label='ple')
                    plt.legend()
                    plt.show()
                    print('hit')
    return


get_most_similar_waveform()
