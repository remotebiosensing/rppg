import multiprocessing
import os

import h5py
import numpy as np
from torch.utils.data import random_split

from utils.image_preprocess import Deepphys_preprocess_Video, PhysNet_preprocess_Video, RTNet_preprocess_Video, \
    GCN_preprocess_Video, Axis_preprocess_Video, RhythmNet_preprocess_Video
from utils.seq_preprocess import PPNet_preprocess_Mat
from utils.text_preprocess import Deepphys_preprocess_Label, PhysNet_preprocess_Label, GCN_preprocess_Label, \
    Axis_preprocess_Label, RhythmNet_preprocess_Label


def dataset_split(dataset, ratio):
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


def preprocessing(save_root_path: str = "/media/hdd1/dy_dataset/",
                  model_name: str = "DeepPhys",
                  data_root_path: str = "/media/hdd1/",
                  dataset_name: str = "UBFC",
                  train_ratio: float = 0.8,
                  face_detect_algorithm: int = 1,
                  divide_flag: bool = True,
                  fixed_position: bool = True,
                  time_length: int = 32,
                  img_size: int = 32,
                  log_flag: bool = True):
    """
    :param save_root_path: save file destination path
    :param model_name: select preprocessing method
    :param data_root_path: data set root path
    :param dataset_name: data set name(ex. UBFC, COFACE)
    :param train_ratio: data split [ train ratio : 1 - train ratio]
    :param face_detect_algorithm: select face_Detect algorithm
    :param divide_flag : True : divide by number, False : divide by subject
    :param fixed_position: True : fixed position, False : face tracking
    :return:
    """

    if log_flag:
        print("=========== preprocessing() in " + os.path.basename(__file__))

    split_flag = True
    dataset_root_path = data_root_path + dataset_name

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    if dataset_name == "V4V":
        dataset_root_path = dataset_root_path + "/train_val/data"
        data_list = [data for data in os.listdir(dataset_root_path)]
        vid_name = "/video.mkv"
        ground_truth_name = "/label.txt"
        print(data_list)
    elif dataset_name == "UBFC":
        data_list = [data for data in os.listdir(dataset_root_path) if data.__contains__("subject")]
        vid_name = "/vid.avi"
        ground_truth_name = "/ground_truth.txt"
    elif dataset_name == "cuff_less_blood_pressure":
        data_list = [data for data in os.listdir(dataset_root_path) if data.__contains__("part")]
    elif dataset_name == "VIPL_HR":
        data_dir = "/data"
        person_data_path = dataset_root_path + data_dir
        # source1/2/3
        # /data/pxx/vxx/sourcex/

        data_list = []
        person_list = [data for data in os.listdir(person_data_path) if data.__contains__("p")]
        for person in person_list:
            v_list = [v for v in os.listdir(person_data_path + "/" + person) if v.__contains__(v)]
            for v in v_list:
                source_list = [source for source in os.listdir(person_data_path + "/" + person + "/" + v) if
                               not source.__contains__("4") and not source.__contains__("2")]
                for source in source_list:
                    tmp = data_dir + "/" + person + "/" + v + "/" + source
                    data_list.append(tmp)

        vid_name = "/video.avi"
        ground_truth_name = "/wave.csv"

        print(person_list)
    elif dataset_name.__contains__("cohface"):
        dataset_root_path = data_root_path + "cohface"
        protocol = dataset_root_path + "/" + "protocols/"
        if dataset_name.__contains__("all"):
            protocol += "all/all.txt"
        elif dataset_name.__contains__("clean"):
            protocol += "clean/all.txt"
        elif dataset_name.__contains__("natural"):
            protocol += "natural/all.txt"
        f = open(protocol, 'r')
        data_list = f.readlines()
        data_list = [path.replace("data\n", "") for path in data_list]
        f.close()
        vid_name = "data.mkv"
        ground_truth_name = "data.hdf5"
    elif dataset_name.__contains__("PURE"):
        data_list = os.listdir(dataset_root_path)
        vid_name = "/png"
        ground_truth_name = "/json"
    process = []

    # multiprocessing
    if not split_flag:
        for index, data_path in enumerate(data_list):
            proc = multiprocessing.Process(target=preprocess_Dataset,
                                           args=(dataset_root_path + "/" + data_path, vid_name, ground_truth_name,
                                                 face_detect_algorithm, divide_flag, fixed_position, time_length, model_name, img_size, return_dict))
            process.append(proc)
            proc.start()

        for proc in process:
            proc.join()
    else:
        # loop = len(data_list) // 5
        loop = 20

        for i in range(loop):
            for index, data_path in enumerate(data_list[i * 5:(i + 1) * 5]):
                proc = multiprocessing.Process(target=preprocess_Dataset,
                                               args=(dataset_root_path + "/" + data_path, vid_name, ground_truth_name,
                                                     face_detect_algorithm, divide_flag, fixed_position, time_length,
                                                     model_name, img_size, return_dict))
                # flag 0 : pass
                # flag 1 : detect face
                # flag 2 : remove nose
                process.append(proc)
                proc.start()

            for proc in process:
                proc.join()

    train = int(len(return_dict.keys()) * train_ratio)  # split dataset
    train_file_path = save_root_path + model_name + "_" + dataset_name + "_train.hdf5"
    test_file_path = save_root_path + model_name + "_" + dataset_name + "_test.hdf5"
    dt = h5py.special_dtype(vlen=np.float32)
    train_file = h5py.File(save_root_path + model_name + "_" + dataset_name + "_train.hdf5", "w")

    if model_name in ["DeepPhys", "PhysNet", "PhysNet_LSTM"]:

        for index, data_path in enumerate(return_dict.keys()[:train]):
            dset = train_file.create_group(data_path)
            dset['preprocessed_video'] = return_dict[data_path]['preprocessed_video']
            dset['preprocessed_label'] = return_dict[data_path]['preprocessed_label']
            dset['preprocessed_hr'] = return_dict[data_path]['preprocessed_hr']

        train_file.close()

        test_file = h5py.File(save_root_path + model_name + "_" + dataset_name + "_test.hdf5", "w")
        for index, data_path in enumerate(return_dict.keys()[train:]):
            dset = test_file.create_group(data_path)
            dset['preprocessed_video'] = return_dict[data_path]['preprocessed_video']
            dset['preprocessed_label'] = return_dict[data_path]['preprocessed_label']
            dset['preprocessed_hr'] = return_dict[data_path]['preprocessed_hr']
        test_file.close()

    elif model_name in ["PPNet"]:

        for index, data_path in enumerate(return_dict.keys()[:train]):
            dset = train_file.create_group(data_path)
            dset['ppg'] = return_dict[data_path]['ppg']
            dset['sbp'] = return_dict[data_path]['sbp']
            dset['dbp'] = return_dict[data_path]['dbp']
            dset['hr'] = return_dict[data_path]['hr']
        train_file.close()

        test_file = h5py.File(save_root_path + model_name + "_" + dataset_name + "_test.hdf5", "w")
        for index, data_path in enumerate(return_dict.keys()[train:]):
            dset = test_file.create_group(data_path)
            dset['ppg'] = return_dict[data_path]['ppg']
            dset['sbp'] = return_dict[data_path]['sbp']
            dset['dbp'] = return_dict[data_path]['dbp']
            dset['hr'] = return_dict[data_path]['hr']
        test_file.close()

    elif model_name in ["GCN"]:
        for index, data_path in enumerate(return_dict.keys()[:train]):
            dset = train_file.create_group(data_path)
            dset['preprocessed_video'] = return_dict[data_path]['preprocessed_video']
            dset['preprocessed_label'] = return_dict[data_path]['preprocessed_label']
        train_file.close()

        test_file = h5py.File(save_root_path + model_name + "_" + dataset_name + "_test.hdf5", "w")
        for index, data_path in enumerate(return_dict.keys()[train:]):
            dset = test_file.create_group(data_path)
            dset['preprocessed_video'] = return_dict[data_path]['preprocessed_video']
            dset['preprocessed_label'] = return_dict[data_path]['preprocessed_label']
        test_file.close()
    elif model_name in ["AxisNet"]:
        for index, data_path in enumerate(return_dict.keys()[:train]):
            dset = train_file.create_group(data_path)
            dset['preprocessed_video'] = return_dict[data_path]['preprocessed_video']
            dset['preprocessed_label'] = return_dict[data_path]['preprocessed_label']
            dset['preprocessed_ptt'] = return_dict[data_path]['preprocessed_ptt']
        train_file.close()

        test_file = h5py.File(save_root_path + model_name + "_" + dataset_name + "_test.hdf5", "w")
        for index, data_path in enumerate(return_dict.keys()[train:]):
            dset = test_file.create_group(data_path)
            dset['preprocessed_video'] = return_dict[data_path]['preprocessed_video']
            dset['preprocessed_label'] = return_dict[data_path]['preprocessed_label']
            dset['preprocessed_ptt'] = return_dict[data_path]['preprocessed_ptt']
        test_file.close()
    elif model_name in ["RhythmNet"]:
        for index, data_path in enumerate(return_dict.keys()[:train]):
            dset = train_file.create_group(data_path)
            dset['preprocessed_video'] = return_dict[data_path]['preprocessed_video']
            dset['preprocessed_label'] = return_dict[data_path]['preprocessed_label']
        train_file.close()

        test_file = h5py.File(save_root_path + model_name + "_" + dataset_name + "_test.hdf5", "w")
        for index, data_path in enumerate(return_dict.keys()[train:]):
            dset = test_file.create_group(data_path)
            dset['preprocessed_video'] = return_dict[data_path]['preprocessed_video']
            dset['preprocessed_label'] = return_dict[data_path]['preprocessed_label']
        test_file.close()


def preprocess_Dataset(path, vid_name, ground_truth_name, face_detect_algorithm, divide_flag, fixed_position,
                       time_length, model_name, img_size, return_dict):
    """
    :param path: dataset path
    :param flag: face detect flag
    :param model_name: select preprocessing method
    :param return_dict: : preprocessed image, label

    """

    # Video Based
    if model_name == "DeepPhys":
        rst, preprocessed_video = Deepphys_preprocess_Video(path + vid_name, face_detect_algorithm, divide_flag,
                                                            fixed_position, time_length, img_size)
    elif model_name == "PhysNet" or model_name == "PhysNet_LSTM":
        rst, preprocessed_video = PhysNet_preprocess_Video(path + vid_name, face_detect_algorithm, divide_flag,
                                                           fixed_position, time_length, img_size)
    elif model_name == "RTNet":
        rst, preprocessed_video = RTNet_preprocess_Video(path + vid_name, face_detect_algorithm, divide_flag,
                                                         fixed_position, time_length, img_size)
    elif model_name == "PPNet":  # Sequence data based
        ppg, sbp, dbp, hr = PPNet_preprocess_Mat(path)
    elif model_name == "GCN":
        rst, preprocessed_video, sliding_window_stride = GCN_preprocess_Video(path + vid_name, face_detect_algorithm,
                                                                              divide_flag, fixed_position, time_length, img_size)
    elif model_name == "AxisNet":
        rst, preprocessed_video, sliding_window_stride, num_frames, stacked_ptts = Axis_preprocess_Video(
            path + vid_name, face_detect_algorithm, divide_flag, fixed_position, time_length, img_size)
    elif model_name == "RhythmNet":
        rst, preprocessed_video = RhythmNet_preprocess_Video(path + vid_name, face_detect_algorithm, divide_flag,
                                                             fixed_position, time_length)

    # rst,bvp,sliding,frames,ptt
    if model_name in ["DeepPhys", "MTTS", "PhysNet", "PhysNet_LSTM", "RhythmNet"]:  # can't detect face
        if not rst:
            return

    if model_name == "DeepPhys":
        preprocessed_label = Deepphys_preprocess_Label(path + ground_truth_name)
    elif model_name == "PhysNet" or model_name == "PhysNet_LSTM":
        preprocessed_label, preprocessed_hr = PhysNet_preprocess_Label(path + ground_truth_name)
    elif model_name == "GCN":
        preprocessed_label = GCN_preprocess_Label(path + ground_truth_name, sliding_window_stride)
    elif model_name == "AxisNet":
        preprocessed_label = Axis_preprocess_Label(path + ground_truth_name, sliding_window_stride, num_frames)
    elif model_name == "RhythmNet":
        preprocessed_label = RhythmNet_preprocess_Label(path + ground_truth_name, time_length)

    # ppg, sbp, dbp, hr
    if model_name in ["DeepPhys", "PhysNet", "PhysNet_LSTM"]:
        return_dict[path.replace('/', '')] = {'preprocessed_video': preprocessed_video,
                                              'preprocessed_label': preprocessed_label,
                                              'preprocessed_hr': preprocessed_hr}
    elif model_name in ["PPNet"]:
        return_dict[path.replace('/', '')] = {'ppg': ppg, 'sbp': sbp, 'dbp': dbp, 'hr': hr}
    elif model_name in ["GCN"]:
        return_dict[path.replace('/', '')] = {'preprocessed_video': preprocessed_video,
                                              'preprocessed_label': preprocessed_label}
    elif model_name in ["AxisNet"]:
        return_dict[path.replace('/', '')] = {'preprocessed_video': preprocessed_video,
                                              'preprocessed_ptt': stacked_ptts,
                                              'preprocessed_label': preprocessed_label}
    elif model_name in ["RhythmNet"]:
        return_dict[path.replace('/', '')] = {'preprocessed_video': preprocessed_video,
                                              'preprocessed_label': preprocessed_label}
        # 'preprocessed_graph': saved_graph}

if __name__ == '__main__':
    preprocessing(save_root_path = "/home/najy/dy/dataset/",
                  model_name = "RhythmNet",
                  data_root_path = "/",
                  dataset_name = "V4V",
                  train_ratio = 0.8,
                  face_detect_algorithm = 1,
                  divide_flag = True,
                  fixed_position = True,
                  time_length= 300,
                  img_size = 36,
                  log_flag = True)