import os

import cv2
import h5py
import numpy as np
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data.sampler import Sampler

from rppg.datasets.APNETv2Dataset import APNETv2Dataset
from rppg.datasets.DeepPhysDataset import DeepPhysDataset
from rppg.datasets.ETArPPGNetDataset import ETArPPGNetDataset
from rppg.datasets.PhysNetDataset import PhysNetDataset
from rppg.datasets.RhythmNetDataset import RhythmNetDataset
from rppg.datasets.VitamonDataset import VitamonDataset
from rppg.datasets.EfficientPhysDataset import EfficientPhysDataset
from rppg.utils.funcs import detrend


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
        return datasets[0], datasets[1], datasets[2]
    elif ratio.__len__() == 2:
        train_len = int(np.floor(dataset_len * ratio[0]))
        val_len = dataset_len - train_len
        datasets = random_split(dataset, [train_len, val_len])
        return datasets[0], datasets[1]


def data_loader(datasets, fit_cfg):
    model_type = fit_cfg.type
    train_batch_size = fit_cfg.train.batch_size
    test_batch_size = fit_cfg.test.batch_size
    time_length = fit_cfg.time_length
    shuffle = fit_cfg.train.shuffle
    meta = fit_cfg.train.meta.flag

    if meta:
        data_loader = []
        for dataset in datasets:
            dataset_len = len(dataset)
            support_len = int(np.floor(dataset_len * 0.8))
            query_len = dataset_len - support_len
            support_dataset, query_dataset = random_split(dataset, [support_len, query_len])
            support_loader = DataLoader(support_dataset, batch_size=train_batch_size, shuffle=False)
            query_loader = DataLoader(query_dataset, batch_size=train_batch_size, shuffle=False)
            data_loader.append([support_loader, query_loader])
        return data_loader

    test_loader = []
    if datasets.__len__() == 3 or datasets.__len__() == 2:
        if model_type == 'DIFF':
            total_len_train = datasets[0].__len__()
            total_len_validation = datasets[1].__len__()
            idx_train = np.arange(total_len_train)
            idx_validation = np.arange(total_len_validation)
            if shuffle:
                idx_train = idx_train.reshape(-1, time_length)
                idx_train = np.random.permutation(idx_train)
                idx_train = idx_train.reshape(-1)
                idx_validation = idx_validation.reshape(-1, time_length)
                idx_validation = np.random.permutation(idx_validation)
                idx_validation = idx_validation.reshape(-1)
                shuffle = False
            sampler_train = ClipSampler(idx_train)
            sampler_validation = ClipSampler(idx_validation)

            train_loader = DataLoader(datasets[0], batch_size=(train_batch_size * time_length),
                                      sampler=sampler_train, shuffle=shuffle)
            validation_loader = DataLoader(datasets[1], batch_size=(train_batch_size * time_length),
                                           sampler=sampler_validation, shuffle=shuffle)
            if datasets.__len__() == 2:
                return [train_loader, validation_loader]
            elif datasets.__len__() == 3:
                for dataset in datasets[2]:
                    test_loader.append(DataLoader(dataset, (test_batch_size * time_length), shuffle=False))
                return [train_loader, validation_loader, test_loader]
        # elif fit_type == 'CONT':
        else:
            train_loader = DataLoader(datasets[0], batch_size=train_batch_size, shuffle=shuffle)
            validation_loader = DataLoader(datasets[1], batch_size=train_batch_size, shuffle=shuffle)
            if datasets.__len__() == 2:
                return [train_loader, validation_loader]
            elif datasets.__len__() == 3:
                for dataset in datasets[2]:
                    test_loader.append(DataLoader(dataset, test_batch_size, shuffle=False))
                return [train_loader, validation_loader, test_loader]

    elif datasets.__len__() == 1:
        if model_type == 'DIFF':
            for dataset in datasets[0]:
                test_loader.append(DataLoader(dataset, (test_batch_size * time_length), shuffle=False))
            return [test_loader]
        else:
            for dataset in datasets[0]:
                test_loader.append(DataLoader(dataset, test_batch_size, shuffle=False))
            return [test_loader]


def dataset_loader(fit_cfg, pre_cfg):
    model_name = fit_cfg.model
    dataset_name = [fit_cfg.train.dataset, fit_cfg.test.dataset]
    time_length = fit_cfg.time_length
    overlap_interval = fit_cfg.overlap_interval
    img_size = fit_cfg.img_size
    train_flag = fit_cfg.train_flag
    eval_flag = fit_cfg.eval_flag
    debug_flag = fit_cfg.debug_flag
    meta = fit_cfg.train.meta.flag

    save_root_path = pre_cfg.dataset_path
    preprocessed_img_size = str(pre_cfg.dataset.image_size)
    if model_name in ["DeepPhys", "TSCAN", "MTTS", "BigSmall"]:
        model_type = 'DIFF'
    elif model_name in ['GREEN','POS','CHROM','LGI','PBV','SSR','PCA','ICA']:
        model_type = 'CONT_RAW'
    else:
        model_type = 'CONT'

    if dataset_name[0] == dataset_name[1]:
        root_file_path = save_root_path + dataset_name[0] + "/" + model_type+"_"+preprocessed_img_size

        path = get_all_files_in_path(root_file_path)
        if debug_flag:
            path = path[:10]
        path_len = len(path)
        # for test

        test_len = 0
        if eval_flag and train_flag:
            test_len = int(np.floor(path_len * 0.1))
            eval_path = path[-test_len:]
        else:
            eval_path = path

        if train_flag:
            train_len = int(np.floor(path_len * 0.8))
            train_path = path[:train_len]
            val_path = path[train_len:]

    elif dataset_name[0] != dataset_name[1]:

        root_file_path = save_root_path + dataset_name[0] + "/" + model_type.split('_')[0]+"_"+preprocessed_img_size
        if not os.path.exists(root_file_path):
            raise FileExistsError("There is no dataset in the path : ", root_file_path)

        path = get_all_files_in_path(root_file_path)
        if debug_flag:
            path = path[:3]
        path_len = len(path)

        if train_flag:
            train_len = int(np.floor(path_len * 0.9))
            train_path = path[:train_len]
            val_path = path[train_len:]

        if eval_flag:
            root_file_path = save_root_path + dataset_name[1] + "/" + model_type.split('_')[0]+"_"+preprocessed_img_size
            if not os.path.exists(root_file_path):
                raise FileExistsError("There is no dataset in the path : ", root_file_path)
            path = get_all_files_in_path(root_file_path)[:]
            if debug_flag:
                path = path[:3]
            eval_path = path

    idx = 0
    dataset_memory = 0

    if meta:
        dataset = []
        # path = ["/hdd/hdd1/dataset/rppg/preprocessed/VIPL_HR/CONT/p1/v1/source1.hdf5"]
        for file_name in path:
            video_data = []
            label_data = []
            if not os.path.isfile(file_name):
                print("Stop at ", idx)
                break
            else:
                file_size = os.path.getsize(file_name)
                print("file size : ", file_size / 1024 / 1024, 'MB')
                dataset_memory += file_size

            file = h5py.File(file_name)
            h5_tree(file)

            if model_name in ["PhysNet", "PhysNet_LSTM", "GCN"]:
                start = 0
                end = time_length
                # label = detrend(file['preprocessed_label'], 100)
                label = file['preprocessed_label']
                num_frame, w, h, c = file['raw_video'][:].shape

                if len(label) != num_frame:
                    label = np.interp(
                        np.linspace(
                            1, len(label), num_frame), np.linspace(
                            1, len(label), len(label)), label)

                if w != img_size:
                    new_shape = (num_frame, img_size, img_size, c)
                    resized_img = np.zeros(new_shape)
                    for i in range(num_frame):
                        # img = file['raw_video'][i] / 255.0
                        img = file['raw_video'][i]
                        resized_img[i] = cv2.resize(img, (img_size, img_size))

                while end <= len(file['raw_video']):
                    if w != img_size:
                        video_chunk = resized_img[start:end]
                    else:
                        video_chunk = file['raw_video'][start:end]
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

                file.close()

                rst_dataset = PhysNetDataset(video_data=np.asarray(video_data),
                                             label_data=np.asarray(label_data),
                                             target_length=time_length)

            elif model_name in ["DeepPhys", "MTTS"]:
                for key in file.keys():
                    rst_dataset.append(DeepPhysDataset(appearance_data=np.asarray(file[key]['raw_video'][:, :, :, -3:]),
                                                       motion_data=np.asarray(file[key]['raw_video'][:, :, :, :3]),
                                                       target=np.asarray(file[key]['preprocessed_label'])))

            dataset.append(rst_dataset)

    else:
        dataset = []
        if train_flag:
            train_dataset = get_dataset(train_path, model_type, model_name, time_length, overlap_interval,
                                        img_size, False)
            dataset.append(train_dataset)
            val_dataset = get_dataset(val_path, model_type, model_name, time_length, 0, img_size, False)
            dataset.append(val_dataset)
        if eval_flag:
            eval_dataset = get_dataset(eval_path, model_type, model_name, time_length, 0, img_size, True)
            dataset.append(eval_dataset)

    return dataset


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


def get_all_files_in_path(path):
    """Get all files in a given path.

    Args:
      path: The path to the directory to search.

    Returns:
      A list of all files in the directory.
    """
    files = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            files.append(os.path.join(dirpath, filename))
    return files


def get_dataset(path, model_type, model_name, time_length, overlap_interval, img_size, eval_flag):
    idx = 0
    round_flag = 0
    rst_dataset = None
    datasets = []

    while True:
        if round_flag == 0:
            if model_type == 'DIFF':
                appearance_data = []
                motion_data = []
                label_data = []
            elif model_type.__contains__('CONT'):
                video_data = []
                label_data = []
                bpm_data = []
                keypoint_data = []
            round_flag = 1
        elif round_flag == 1:
            if idx == len(path):
                break
            file_name = path[idx]
            print(file_name)
            if not os.path.isfile(file_name):
                print("Stop at ", idx)
                break

            idx += 1
            file = h5py.File(file_name)
            h5_tree(file)
            if model_type == 'DIFF':
                num_frame, w, h, c = file['raw_video'][:].shape
                if model_name == "BigSmall":
                    for i in range(num_frame):
                        img = file['raw_video'][i]
                        appearance_data.append(cv2.resize(img[:, :, 3:], (144, 144), interpolation=cv2.INTER_AREA))
                        motion_data.append(cv2.resize(img[:, :, :3], (9, 9), interpolation=cv2.INTER_AREA))

                elif w != img_size:
                    new_shape = (num_frame, img_size, img_size, c)
                    resized_img = np.zeros(new_shape)
                    for i in range(num_frame):
                        img = file['raw_video'][i]
                        resized_img[i] = cv2.resize(img, (img_size, img_size))
                    appearance_data.extend(resized_img[:, :, :, -3:])
                    motion_data.extend(resized_img[:, :, :, :3])
                else:
                    appearance_data.extend(file['raw_video'][:, :, :, -3:])
                    motion_data.extend(file['raw_video'][:, :, :, :3])

                temp_label = file['preprocessed_label']
                # resample label data
                if len(temp_label) != num_frame:
                    print('-----Resampling label data-----')
                    temp_label = np.interp(
                        np.linspace(
                            1, len(temp_label), num_frame), np.linspace(
                            1, len(temp_label), len(temp_label)), temp_label)
                label_data.extend(temp_label)

                num_frame = (num_frame // time_length) * time_length
                appearance_data = appearance_data[:num_frame]
                motion_data = motion_data[:num_frame]
                label_data = label_data[:num_frame]

            elif model_name in ["APNETv2"]:
                start = 0
                end = time_length
                label = detrend(file['preprocessed_label'], 100)

                while end <= len(file['raw_video']):
                    video_chunk = file['raw_video'][start:end]
                    video_data.append(video_chunk)
                    keypoint_data.append(file['keypoint'][start:end])
                    tmp_label = label[start:end]

                    tmp_label = np.around(normalize(tmp_label, 0, 1), 2)
                    label_data.append(tmp_label)
                    # video_chunks.append(video_chunk)
                    start += time_length - overlap_interval
                    end += time_length - overlap_interval

            elif model_name in ["EfficientPhys"]:
                label = file['preprocessed_label']
                diff_norm_label = np.diff(label, axis=0)
                diff_norm_label /= np.std(diff_norm_label)
                diff_norm_label = np.array(diff_norm_label)
                diff_norm_label[np.isnan(diff_norm_label)] = 0

                num_frame, w, h, c = file['raw_video'][:].shape
                if w != img_size and h != img_size:
                    new_shape = (num_frame, img_size, img_size, c)
                    resized_img = np.zeros(new_shape, dtype=np.float32)
                    for i in range(num_frame):
                        img = file['raw_video'][i]  # / 255.
                        resized_img[i] = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
                    diff_video = np.diff(resized_img, axis=0)
                else:
                    diff_video = np.diff(file['raw_video'][:], axis=0)

                num_frame = ((num_frame - 1) // time_length) * time_length
                label_data.extend(diff_norm_label[:num_frame])
                video_data.extend(diff_video[:num_frame])

            else:
                start = 0
                end = time_length
                # label = detrend(file['preprocessed_label'], 100)
                label = file['preprocessed_label']
                num_frame, w, h, c = file['raw_video'][:].shape

                if len(label) != num_frame:
                    label = np.interp(
                        np.linspace(
                            1, len(label), num_frame), np.linspace(
                            1, len(label), len(label)), label)

                if w != img_size and h != img_size:
                    new_shape = (num_frame, img_size, img_size, c)
                    if model_type.__contains__('RAW'):
                        resized_img = np.zeros(new_shape, dtype=np.uint8)
                        for i in range(num_frame):
                            img = file['raw_video'][i] * 255
                            w, h, c = img.shape
                            w_m, h_m = w - round(w * 2/3), h - round(h * 2/3)
                            img = cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_BGR2RGB)
                            resized_img[i] = cv2.resize(img[w_m//2:-w_m//2,h_m//2:-h_m//2], (img_size, img_size), interpolation=cv2.INTER_AREA)
                    else:
                        resized_img = np.zeros(new_shape, dtype=np.float32)
                        for i in range(num_frame):
                            img = file['raw_video'][i]
                            resized_img[i] = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

                while end <= len(file['raw_video']):
                    if w != img_size:
                        video_chunk = resized_img[start:end]
                    else:
                        video_chunk = file['raw_video'][start:end]
                    if not model_type.__contains__('RAW'):
                        video_chunk = (video_chunk - np.mean(video_chunk)) / np.std(video_chunk)
                    # video_chunk = int(video_chunk*)
                    video_data.append(video_chunk)
                    tmp_label = label[start:end]

                    # tmp_label = np.around(normalize(tmp_label, 0, 1), 2)
                    label_data.append(tmp_label)
                    # video_chunks.append(video_chunk)
                    start += time_length - overlap_interval
                    end += time_length - overlap_interval

            file.close()
            round_flag = 2
        elif round_flag == 2:
            if model_type == 'DIFF':
                dataset = DeepPhysDataset(appearance_data=np.asarray(appearance_data),
                                          motion_data=np.asarray(motion_data),
                                          target=np.asarray(label_data))
            elif model_name in ["APNETv2"]:
                dataset = APNETv2Dataset(video_data=np.asarray(video_data),
                                         keypoint_data=np.asarray(keypoint_data),
                                         label_data=np.asarray(label_data),
                                         target_length=time_length,
                                         img_size=img_size)
            elif model_name in ["EfficientPhys"]:
                dataset = EfficientPhysDataset(video_data=np.asarray(video_data),
                                               label_data=np.asarray(label_data))
            elif model_type.__contains__('CONT'):
                dataset = PhysNetDataset(video_data=np.asarray(video_data),
                                         label_data=np.asarray(label_data),
                                         target_length=time_length)
            # elif model_name in ["RhythmNet"]:
            #     dataset = RhythmNetDataset(st_map_data=np.asarray(st_map_data),
            #                                target_data=np.asarray(target_data))
            elif model_name in ["ETArPPGNet"]:
                dataset = ETArPPGNetDataset(video_data=np.asarray(video_data),
                                            label_data=np.asarray(label_data))

            elif model_name in ["Vitamon", "Vitamon_phase2"]:
                dataset = VitamonDataset(video_data=np.asarray(video_data),
                                         label_data=np.asarray(label_data))
            if not eval_flag:
                datasets = [rst_dataset, dataset]
                rst_dataset = ConcatDataset([dataset for dataset in datasets if dataset is not None])
            else:
                datasets.append(dataset)
            round_flag = 0
    if not eval_flag:
        return rst_dataset
    else:
        return datasets


class ClipSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(self.data_source.tolist())

    def __len__(self):
        return len(self.data_source.tolist())
