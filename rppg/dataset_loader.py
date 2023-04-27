import os

import h5py
import numpy as np
import psutil
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from params import params
from rppg.datasets.APNETv2Dataset import APNETv2Dataset
from rppg.datasets.DeepPhysDataset import DeepPhysDataset
from rppg.datasets.ETArPPGNetDataset import ETArPPGNetDataset
from rppg.datasets.PhysNetDataset import PhysNetDataset
from rppg.datasets.RhythmNetDataset import RhythmNetDataset
from rppg.datasets.VitamonDataset import VitamonDataset
from utils.funcs import detrend


def dataset_split(**kwargs):
    dataset, ratio = kwargs["dataset"], kwargs["ratio"]
    dataset_len = len(dataset)
    if ratio.__len__() == 3:
        train_len = int(np.floor(dataset_len * ratio[0]))
        val_len = int(np.floor(dataset_len * ratio[1]))
        test_len = dataset_len - train_len - val_len

        return random_split(dataset, [train_len, val_len, test_len])
    elif ratio.__len__() == 2:
        train_len = int(np.floor(dataset_len * ratio[0]))
        test_len = dataset_len - train_len
        return random_split(dataset, [train_len, test_len])



def split_data_loader(**kwargs):
    datasets, batch_size, train_shuffle, test_shuffle = kwargs["datasets"], kwargs["batch_size"], kwargs["train_shuffle"], kwargs["test_shuffle"]
    if datasets.__len__() == 3:
        train_loader = DataLoader(datasets[0], batch_size=batch_size, shuffle=train_shuffle)
        validation_loader = DataLoader(datasets[1], batch_size=batch_size, shuffle=test_shuffle)
        test_loader = DataLoader(datasets[2], batch_size=batch_size, shuffle=test_shuffle)
        return [train_loader, validation_loader, test_loader]
    elif datasets.__len__() == 2:
        train_loader = DataLoader(datasets[0], batch_size=batch_size, shuffle=train_shuffle)
        test_loader = DataLoader(datasets[1], batch_size=batch_size, shuffle=test_shuffle)
        return [train_loader, test_loader]


def dataset_loader():
    '''
    :param save_root_path: save file destination path
    :param model_name : model_name
    :param dataset_name: data set name(ex. UBFC, COFACE)
    :param option:[train, test]
    :return: dataset
    '''
    if params.log_flag:
        print("========= dataset_loader() in" + os.path.basename(__file__))

    cnt = 0
    flag = True
    name = params.model



    available_memory = psutil.virtual_memory().available
    print("available_memory : ", available_memory / 1024 / 1024, 'MB')

    root_file_path = params.save_root_path + name + "_" + params.dataset_name + "_" + params.dataset_date +"_"
    idx = 0

    target_memory = 1024*1024*1024*3

    dataset_memory = 0

    round_flag = 0

    rst_dataset = None

    while True:
        if round_flag == 0:
            if params.model in ["DeepPhys", "MTTS"]:
                appearance_data = []
                motion_data = []
                target_data = []
            elif params.model in ["PhysNet", "PhysNet_LSTM", "GCN"]:
                video_data = []
                label_data = []
                bpm_data = []
                keypoint_data = []
            elif params.model in ["TEST"]:
                video_data = []
                keypoint_data = []
                label_data = []
            elif params.model in ["PPNet"]:
                ppg = []
                sbp = []
                dbp = []
                hr = []
            elif params.model in ["RTNet"]:
                face_data = []
                mask_data = []
                target_data = []
            elif params.model in ["AxisNet"]:
                video_data = []
                label_data = []
                ptt_data = []
            elif params.model in ["RhythmNet"]:
                st_map_data = []
                target_data = []
            elif params.model in ["ETArPPGNet"]:
                video_data = []
                label_data = []
            elif params.model in ["Vitamon", "Vitamon_phase2"]:
                video_data = []
                label_data = []
            round_flag = 1
        elif round_flag == 1:
            file_name = root_file_path + str(idx) + ".hdf5"

            if not os.path.isfile(file_name):
                print("Stop at ", idx)
                break
            else:
                file_size = os.path.getsize(file_name)
                print("file size : ", file_size/ 1024 / 1024, 'MB')
                dataset_memory += file_size

            idx += 1
            file = h5py.File(file_name, "r")
            h5_tree(file)
            for key in file.keys():
                if params.model in ["DeepPhys", "MTTS"]:
                    appearance_data.extend(file[key]['preprocessed_video'][:, :, :, -3:])
                    motion_data.extend(file[key]['preprocessed_video'][:, :, :, :3])
                    target_data.extend(file[key]['preprocessed_label'])
                elif params.model in ["TEST"]:
                    start = 0
                    end = params.time_length
                    label =detrend(file[key]['preprocessed_label'],100)

                    while end <= len(file[key]['raw_video']):
                        video_chunk = file[key]['raw_video'][start:end]
                        # min_val = np.min(video_chunk, axis=(0, 1, 2), keepdims=True)
                        # max_val = np.max(video_chunk, axis=(0, 1, 2), keepdims=True)
                        # video_chunk = (video_chunk - min_val) / (max_val - min_val)
                        video_data.append(video_chunk)
                        keypoint_data.append(file[key]['keypoint'][start:end])
                        tmp_label = label[start:end]

                        tmp_label = np.around(normalize(tmp_label,0,1),2)
                        label_data.append(tmp_label)
                        # video_chunks.append(video_chunk)
                        start += params.time_length - params.interval
                        end += params.time_length - params.interval
                elif params.model in ["PhysNet", "PhysNet_LSTM", "GCN"]:
                    if len(file[key]['preprocessed_video']) == len(file[key]['preprocessed_label']):
                        video_data.extend(file[key]['preprocessed_video'])
                        label_data.extend(file[key]['preprocessed_label'])
                elif params.model in ["PPNet"]:
                    ppg.extend(file[key]['ppg'])
                    sbp.extend(file[key]['sbp'])
                    dbp.extend(file[key]['dbp'])
                    hr.extend(file[key]['hr'])
                elif params.model in ["RTNet"]:
                    face_data.extend(file[key]['preprocessed_video'][:, :, :, -3:])
                    mask_data.extend(file[key]['preprocessed_video'][:, :, :, :3])
                    target_data.extend(file[key]['preprocessed_label'])
                elif params.model in ["AxisNet"]:
                    video_data.extend(file[key]['preprocessed_video'])
                    ptt_data.extend(file[key]['preprocessed_ptt'])
                    label_data.extend(file[key]['preprocessed_label'])
                elif params.model in ["RhythmNet"]:
                    st_map_data.extend(file[key]['preprocessed_video'])
                    target_data.extend(file[key]['preprocessed_label'])
                elif params.model in ["ETArPPGNet"]:
                    video_data.extend(file[key]['preprocessed_video'])
                    label_data.extend(file[key]['preprocessed_label'])
                elif params.model in ["Vitamon", "Vitamon_phase2"]:
                    video_data.extend(file[key]['preprocessed_video'])
                    label_data.extend(file[key]['preprocessed_label'])
            file.close()
            if dataset_memory > target_memory:
                round_flag = 2
        elif round_flag == 2:
            if params.model in ["DeepPhys", "MTTS"]:
                dataset = DeepPhysDataset(appearance_data=np.asarray(appearance_data),
                                          motion_data=np.asarray(motion_data),
                                          target=np.asarray(target_data))
            elif params.model in ["TEST"]:
                dataset = APNETv2Dataset(video_data=np.asarray(video_data),
                                      keypoint_data=np.asarray(keypoint_data),
                                      label_data=np.asarray(label_data),
                                      target_length=params.time_length)

            elif params.model in ["PhysNet", "PhysNet_LSTM", "GCN"]:
                dataset = PhysNetDataset(video_data=np.asarray(video_data),
                                         label_data=np.asarray(label_data))
            elif params.model in ["RhythmNet"]:
                dataset = RhythmNetDataset(st_map_data=np.asarray(st_map_data),
                                           target_data=np.asarray(target_data))
            elif params.model in ["ETArPPGNet"]:
                dataset = ETArPPGNetDataset(video_data=np.asarray(video_data),
                                            label_data=np.asarray(label_data))

            elif params.model in ["Vitamon","Vitamon_phase2"]:
                dataset = VitamonDataset(video_data=np.asarray(video_data),
                                            label_data=np.asarray(label_data))
            datasets = [rst_dataset, dataset]
            rst_dataset = ConcatDataset([dataset for dataset in datasets if dataset is not None])
            round_flag = 0


    return rst_dataset

def h5_tree(val, pre=''):
    items = len(val)
    for key, val in val.items():
        items -= 1
        if items == 0:
            # the last item
            if type(val) == h5py._hl.group.Group:
                print(pre + '└── ' + key)
                h5_tree(val, pre+'    ')
            else:
                print(pre + '└── ' + key + ' (%d)' % len(val))
        else:
            if type(val) == h5py._hl.group.Group:
                print(pre + '├── ' + key)
                h5_tree(val, pre+'│   ')
            else:
                print(pre + '├── ' + key + ' (%d)' % len(val))


def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        tmp = ((( i - min(arr)*diff)/diff_arr)) + t_min
        norm_arr.append(tmp)

    return norm_arr