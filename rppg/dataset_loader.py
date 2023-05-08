import os

import h5py
import numpy as np
import psutil
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from rppg.datasets.APNETv2Dataset import APNETv2Dataset
from rppg.datasets.DeepPhysDataset import DeepPhysDataset
from rppg.datasets.ETArPPGNetDataset import ETArPPGNetDataset
from rppg.datasets.PhysNetDataset import PhysNetDataset
from rppg.datasets.RhythmNetDataset import RhythmNetDataset
from rppg.datasets.VitamonDataset import VitamonDataset
from rppg.datasets.APNETv3Dataset import APNETv3Dataset
from rppg.utils.funcs import detrend

import cv2


def dataset_split(
        dataset,
        ratio
):
    dataset_len = len(dataset)
    if ratio.__len__() == 3:
        train_len = int(np.floor(dataset_len * ratio[0]))
        val_len = int(np.floor(dataset_len * ratio[1]))
        test_len = dataset_len - train_len - val_len
        datasets = random_split(dataset, [train_len, val_len, test_len])
        return datasets[0], datasets[1], datasets[2],
    elif ratio.__len__() == 2:
        train_len = int(np.floor(dataset_len * ratio[0]))
        val_len = dataset_len - train_len
        datasets = random_split(dataset, [train_len, val_len])
        return datasets[0], datasets[1]


def data_loader(
        datasets,
        batch_sizes,
        shuffles,
        meta=False
):
    if meta:
        data_loader = []
        for dataset in datasets:
            data_loader.append(DataLoader(dataset,batch_sizes[0],shuffles[0]))
        return data_loader

    if datasets.__len__() == 3:
        train_loader = DataLoader(datasets[0], batch_size=batch_sizes[0], shuffle=shuffles[0])
        validation_loader = DataLoader(datasets[1], batch_size=batch_sizes[1], shuffle=shuffles[1])
        test_loader = DataLoader(datasets[2], batch_size=batch_sizes[2], shuffle=shuffles[2])
        return [train_loader, validation_loader, test_loader]
    elif datasets.__len__() == 2:
        train_loader = DataLoader(datasets[0], batch_size=batch_sizes[0], shuffle=shuffles[0])
        test_loader = DataLoader(datasets[1], batch_size=batch_sizes[1], shuffle=shuffles[1])
        return [train_loader, test_loader]
    elif datasets.__len__() == 1:
        data_loader = DataLoader(datasets[0], batch_size=batch_sizes[0], shuffle=shuffles[0])
        return [data_loader]


def dataset_loader(

        save_root_path: str,
        model_name: str,
        dataset_name: str,
        time_length: int,
        overlap_interval: int,
        img_size: int,
        meta = False
):
    available_memory = psutil.virtual_memory().available
    print("available_memory : ", available_memory / 1024 / 1024, 'MB')

    model_type = ''
    if model_name in ["DeepPhys", "MTTS"]:
        model_type = 'DIFF'
    else:
        model_type = 'CONT'

    root_file_path = save_root_path + model_type + "_" + dataset_name + "_"

    idx = 0

    target_memory = 1024 * 1024 * 1024 * 3

    dataset_memory = 0

    round_flag = 0

    rst_dataset = None

    if meta:
        rst_dataset = []
        while True:
            idx += 1
            file_name = root_file_path + str(idx) + ".hdf5"

            if not os.path.isfile(file_name):
                print("Stop at ", idx)
                break
            else:
                file_size = os.path.getsize(file_name)
                print("file size : ", file_size / 1024 / 1024, 'MB')
                dataset_memory += file_size

            file = h5py.File(file_name)
            h5_tree(file)

            for key in file.keys():
                if model_name in ["DeepPhys", "MTTS"]:
                    rst_dataset.append(DeepPhysDataset(appearance_data=np.asarray(file[key]['raw_video'][:, :, :, -3:]),
                                    motion_data=np.asarray(file[key]['raw_video'][:, :, :, :3]),
                                    target=np.asarray(file[key]['preprocessed_label'])))


    else:
        while True:
            if round_flag == 0:
                if model_type == 'DIFF':
                    appearance_data = []
                    motion_data = []
                    target_data = []
                elif model_type == 'CONT':
                    video_data = []
                    label_data = []
                    bpm_data = []
                    keypoint_data = []
                round_flag = 1
            elif round_flag == 1:
                file_name = root_file_path + str(idx) + ".hdf5"

                if not os.path.isfile(file_name):
                    print("Stop at ", idx)
                    break
                else:
                    file_size = os.path.getsize(file_name)
                    print("file size : ", file_size / 1024 / 1024, 'MB')
                    dataset_memory += file_size

                idx += 1
                file = h5py.File(file_name)
                h5_tree(file)
                for key in file.keys():
                    if model_name in ["DeepPhys", "MTTS"]:
                        appearance_data.extend(file[key]['raw_video'][:, :, :, -3:])
                        motion_data.extend(file[key]['raw_video'][:, :, :, :3])
                        target_data.extend(file[key]['preprocessed_label'])
                    elif model_name in ["APNETv2"]:
                        start = 0
                        end = time_length
                        label = detrend(file[key]['preprocessed_label'], 100)

                        while end <= len(file[key]['raw_video']):
                            video_chunk = file[key]['raw_video'][start:end]
                            # min_val = np.min(video_chunk, axis=(0, 1, 2), keepdims=True)
                            # max_val = np.max(video_chunk, axis=(0, 1, 2), keepdims=True)
                            # video_chunk = (video_chunk - min_val) / (max_val - min_val)
                            video_data.append(video_chunk)
                            keypoint_data.append(file[key]['keypoint'][start:end])
                            tmp_label = label[start:end]

                            tmp_label = np.around(normalize(tmp_label, 0, 1), 2)
                            label_data.append(tmp_label)
                            # video_chunks.append(video_chunk)
                            start += time_length - overlap_interval
                            end += time_length - overlap_interval
                    elif model_name in ["PhysNet", "PhysNet_LSTM", "GCN","APNETv3","ContrastPhys"]:
                        start = 0
                        end = time_length
                        label = detrend(file[key]['preprocessed_label'], 100)
                        num_frame, w, h, c = file[key]['raw_video'][:].shape
                        if w != img_size:
                            new_shape = (num_frame, img_size, img_size, c)
                            resized_img = np.zeros(new_shape)
                            for i in range(num_frame):
                                resized_img[i] = cv2.resize(file[key]['raw_video'][i], (img_size, img_size))

                        while end <= len(file[key]['raw_video']):
                            if w != img_size:
                                video_chunk = resized_img[start:end]
                            else:
                                video_chunk = file[key]['raw_video'][start:end]
                            # min_val = np.min(video_chunk, axis=(0, 1, 2), keepdims=True)
                            # max_val = np.max(video_chunk, axis=(0, 1, 2), keepdims=True)
                            # video_chunk = (video_chunk - min_val) / (max_val - min_val)
                            video_data.append(video_chunk)
                            tmp_label = label[start:end]

                            tmp_label = np.around(normalize(tmp_label, 0, 1), 2)
                            label_data.append(tmp_label)
                            # video_chunks.append(video_chunk)
                            start += time_length - overlap_interval
                            end += time_length - overlap_interval

                    elif model_name in ["PPNet"]:
                        ppg.extend(file[key]['ppg'])
                        sbp.extend(file[key]['sbp'])
                        dbp.extend(file[key]['dbp'])
                        hr.extend(file[key]['hr'])
                    elif model_name in ["RTNet"]:
                        face_data.extend(file[key]['preprocessed_video'][:, :, :, -3:])
                        mask_data.extend(file[key]['preprocessed_video'][:, :, :, :3])
                        target_data.extend(file[key]['preprocessed_label'])
                    elif model_name in ["AxisNet"]:
                        video_data.extend(file[key]['preprocessed_video'])
                        ptt_data.extend(file[key]['preprocessed_ptt'])
                        label_data.extend(file[key]['preprocessed_label'])
                    elif model_name in ["RhythmNet"]:
                        st_map_data.extend(file[key]['preprocessed_video'])
                        target_data.extend(file[key]['preprocessed_label'])
                    elif model_name in ["ETArPPGNet"]:
                        video_data.extend(file[key]['preprocessed_video'])
                        label_data.extend(file[key]['preprocessed_label'])
                    elif model_name in ["Vitamon", "Vitamon_phase2"]:
                        video_data.extend(file[key]['preprocessed_video'])
                        label_data.extend(file[key]['preprocessed_label'])
                file.close()
                if dataset_memory > target_memory:
                    round_flag = 2
            elif round_flag == 2:
                if model_name in ["DeepPhys", "MTTS"]:
                    dataset = DeepPhysDataset(appearance_data=np.asarray(appearance_data),
                                              motion_data=np.asarray(motion_data),
                                              target=np.asarray(target_data))
                elif model_name in ["APNETv2"]:
                    dataset = APNETv2Dataset(video_data=np.asarray(video_data),
                                             keypoint_data=np.asarray(keypoint_data),
                                             label_data=np.asarray(label_data),
                                             target_length=time_length,
                                             img_size=img_size)
                elif  model_name in ["APNETv3"]:
                    dataset = APNETv3Dataset(video_data=np.asarray(video_data),
                                             label_data=np.asarray(label_data),
                                             target_length=time_length)

                elif model_name in ["PhysNet", "PhysNet_LSTM", "GCN","ContrastPhys"]:
                    dataset = PhysNetDataset(video_data=np.asarray(video_data),
                                             label_data=np.asarray(label_data),
                                             target_length=time_length)
                elif model_name in ["RhythmNet"]:
                    dataset = RhythmNetDataset(st_map_data=np.asarray(st_map_data),
                                               target_data=np.asarray(target_data))
                elif model_name in ["ETArPPGNet"]:
                    dataset = ETArPPGNetDataset(video_data=np.asarray(video_data),
                                                label_data=np.asarray(label_data))

                elif model_name in ["Vitamon", "Vitamon_phase2"]:
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
                h5_tree(val, pre + '    ')
            else:
                print(pre + '└── ' + key + ' (%d)' % len(val))
        else:
            if type(val) == h5py._hl.group.Group:
                print(pre + '├── ' + key)
                h5_tree(val, pre + '│   ')
            else:
                print(pre + '├── ' + key + ' (%d)' % len(val))


def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        tmp = (((i - min(arr) * diff) / diff_arr)) + t_min
        norm_arr.append(tmp)

    return norm_arr
