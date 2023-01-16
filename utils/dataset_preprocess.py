import multiprocessing
import os

import h5py
import numpy as np
from torch.utils.data import random_split

import datetime

from utils.image_preprocess import video_preprocess
from utils.text_preprocess import label_preprocess

import math

from params import params

from log import log_info_time

import cv2


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


def preprocessing():
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
    if not params.__PREPROCESSING__:
        return


    save_root_path = params.save_root_path
    model_name = params.model
    data_root_path = params.data_root_path
    dataset_name = params.dataset_name
    train_ratio = params.train_ratio
    face_detect_algorithm = params.face_detect_algorithm
    divide_flag = params.divide_flag
    fixed_position = params.fixed_position
    time_length = params.time_length
    img_size = params.img_size
    chunk_size = params.chunk_size

    if params.log_flag:
        print("=========== preprocessing() in " + os.path.basename(__file__))

    split_flag = False
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

    ssl_flag = True

    # multiprocessing

    now = datetime.datetime.now()
    time = now.strftime('%Y-%m-%d')

    chunk_num = math.ceil(len(data_list) / chunk_size)
    for i in range(chunk_num):
        if i == chunk_num - 1:
            break
            chunk_data_list = data_list[i * chunk_size:]
        else:
            chunk_data_list = data_list[i * chunk_size:(i + 1) * chunk_size]

        print("chunk_data_list : ", chunk_data_list)

        chunk_preprocessing(model_name, chunk_data_list, dataset_root_path, vid_name, ground_truth_name,
                            face_detect_algorithm=face_detect_algorithm, divide_flag=divide_flag,
                            fixed_position=fixed_position, time_length=time_length, img_size=img_size,
                            ssl_flag=ssl_flag, idx=i, time=time)
        if i == 0 :
            break


def preprocess_Dataset(model_name, path, vid_name, ground_truth_name, return_dict, **kwargs):
    """
    :param path: dataset path
    :param flag: face detect flag
    :param model_name: select preprocessing method
    :param return_dict: : preprocessed image, label

    """

    preprocessed_label = label_preprocess(model_name=model_name,
                                          path=path + ground_truth_name,
                                          **kwargs)

    rst_dict = video_preprocess(model_name=model_name,
                                path=path + vid_name,
                                **kwargs)
    if not rst_dict["face_detect"]:
        return

    # ppg, sbp, dbp, hr
    if model_name in ["DeepPhys", "PhysNet", "PhysNet_LSTM"]:
        return_dict[path.replace('/', '') + str(kwargs['flip_flag'])] = {
            'preprocessed_video': rst_dict["video_data"],
            'frame_number': rst_dict["frame_number"],
            'flip_arr': rst_dict["flip_arr"],
            'keypoint' : rst_dict["keypoint"],
            'raw_video' : rst_dict["raw_video"],
            'preprocessed_label': preprocessed_label}
    elif model_name in ["TEST"]:
        return_dict[path.replace('/', '')] = {
            'keypoint': rst_dict["keypoint"],
            'raw_video': rst_dict["raw_video"],
            'preprocessed_label': preprocessed_label}
        # 'preprocessed_hr': preprocessed_hr}
    elif model_name in ["PPNet"]:
        return_dict[path.replace('/', '')] = {'ppg': ppg, 'sbp': sbp, 'dbp': dbp, 'hr': hr}
    elif model_name in ["GCN", "RhythmNet", "ETArPPGNet"]:
        return_dict[path.replace('/', '')] = {'preprocessed_video': rst_dict["video_data"],
                                              'preprocessed_label': preprocessed_label}
    elif model_name in ["AxisNet"]:
        return_dict[path.replace('/', '')] = {'preprocessed_video': rst_dict["video_data"],
                                              # 'preprocessed_ptt': stacked_ptts,
                                              'preprocessed_label': preprocessed_label}
    elif model_name in ["Vitamon", "Vitamon_phase2"]:
        return_dict[path.replace('/', '')] = {'preprocessed_video': rst_dict["video_data"],
                                              'preprocessed_label': preprocessed_label}

    # 'preprocessed_graph': saved_graph}


def chunk_preprocessing(model_name, data_list, dataset_root_path, vid_name, ground_truth_name, **kwargs):
    process = []
    face_detect_algorithm = kwargs['face_detect_algorithm']
    divide_flag = kwargs['divide_flag']
    fixed_position = kwargs['fixed_position']
    time_length = kwargs['time_length']
    img_size = kwargs['img_size']
    ssl_flag = kwargs['ssl_flag']
    time = kwargs['time']
    idx = kwargs['idx']

    save_root_path = "/media/hdd1/dy/dataset/"
    dataset_name = "UBFC"

    if ssl_flag:
        loop_range = 4
    else:
        loop_range = 1

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    for index, data_path in enumerate(data_list):
        for i in range(loop_range):
            proc = multiprocessing.Process(target=preprocess_Dataset,
                                           args=(
                                               model_name, dataset_root_path + "/" + data_path, vid_name,
                                               ground_truth_name,
                                               return_dict)
                                           , kwargs={"face_detect_algorithm": face_detect_algorithm,
                                                     "divide_flag": divide_flag,
                                                     "fixed_position": fixed_position,
                                                     "time_length": time_length,
                                                     "img_size": img_size,
                                                     "flip_flag": i})

            process.append(proc)
            proc.start()

    print(len(process))
    for proc in process:
        proc.join()


    dataset_path = h5py.File(save_root_path + model_name + "_" + dataset_name + "_" + time + "_" + str(idx) + ".hdf5",
                             "w")

    if model_name in ["DeepPhys", "PhysNet", "PhysNet_LSTM"]:
        for index, data_path in enumerate(return_dict.keys()):
            print(index)
            dset = dataset_path.create_group(data_path)
            dset['preprocessed_video'] = return_dict[data_path]['preprocessed_video']
            dset['frame_number'] = return_dict[data_path]['frame_number']
            dset['preprocessed_label'] = return_dict[data_path]['preprocessed_label'][0]
            dset['preprocessed_hr'] = return_dict[data_path]['preprocessed_label'][1]
            dset['flip_arr'] = return_dict[data_path]['flip_arr']
            dset['keypoint'] = return_dict[data_path]['keypoint']
            dset['raw_video'] = return_dict[data_path]['raw_video']
    elif model_name in ["PPNet"]:
        for index, data_path in enumerate(return_dict.keys()):
            dset = dataset_path.create_group(data_path)
            dset['ppg'] = return_dict[data_path]['ppg']
            dset['sbp'] = return_dict[data_path]['sbp']
            dset['dbp'] = return_dict[data_path]['dbp']
            dset['hr'] = return_dict[data_path]['hr']
    elif model_name in ["GCN", "RhythmNet", "ETArPPGNet"]:
        for index, data_path in enumerate(return_dict.keys()):
            dset = dataset_path.create_group(data_path)
            dset['preprocessed_video'] = return_dict[data_path]['preprocessed_video']
            dset['preprocessed_label'] = return_dict[data_path]['preprocessed_label']
    elif model_name in ["AxisNet"]:
        for index, data_path in enumerate(return_dict.keys()):
            dset = dataset_path.create_group(data_path)
            dset['preprocessed_video'] = return_dict[data_path]['preprocessed_video']
            dset['preprocessed_label'] = return_dict[data_path]['preprocessed_label']
            dset['preprocessed_ptt'] = return_dict[data_path]['preprocessed_ptt']
    elif model_name in ["Vitamon", "Vitamon_phase2"]:
        for index, data_path in enumerate(return_dict.keys()):
            dset = dataset_path.create_group(data_path)
            dset['preprocessed_video'] = return_dict[data_path]['preprocessed_video']
            dset['preprocessed_label'] = return_dict[data_path]['preprocessed_label']
    dataset_path.close()

    manager.shutdown()
