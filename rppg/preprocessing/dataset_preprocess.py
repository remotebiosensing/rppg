import multiprocessing
import os

import h5py
import numpy as np
from torch.utils.data import random_split

import datetime

from rppg.preprocessing.image_preprocess import video_preprocess
from rppg.preprocessing.text_preprocess import label_preprocess

import math

from params import params



def preprocessing(
        data_root_path,
        preprocess_cfg,
        dataset_path
):
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

    chunk_size = preprocess_cfg.chunk_size
    dataset_path = dataset_path
    manager = multiprocessing.Manager()
    for dataset in preprocess_cfg.datasets:
        dataset_name = dataset['name']
        if dataset['type'] == 'continuous':
            preprocess_type = 'CONT'
        else:
            preprocess_type = 'DIFF'
        face_detect_algorithm = dataset['face_detect_algorithm']
        fixed_position = dataset["fixed_position"]
        img_size = dataset['image_size']


        dataset_root_path = data_root_path + dataset_name

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
                    source_list = [source for source in os.listdir(person_data_path + "/" + person + "/" + v)]
                    for source in source_list:
                        tmp = data_dir + "/" + person + "/" + v + "/" + source
                        if len(os.listdir(dataset_root_path+tmp)) == 5 and source == 'source1':
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
        chunk_num = math.ceil(len(data_list) / chunk_size)
        for i in range(chunk_num):
            if i == chunk_num - 1:
                break
                chunk_data_list = data_list[i * chunk_size:]
            else:
                chunk_data_list = data_list[i * chunk_size:(i + 1) * chunk_size]

            print("chunk_data_list : ", chunk_data_list)

            chunk_preprocessing(preprocess_type, chunk_data_list, dataset_root_path, vid_name, ground_truth_name,
                                dataset_name,
                                dataset_path,
                                face_detect_algorithm=face_detect_algorithm,
                                fixed_position=fixed_position, img_size=img_size,
                                chunk_size=chunk_size, idx=i)




def preprocess_Dataset(preprocess_type, path, vid_name, ground_truth_name, return_dict, **kwargs):
    """
    :param path: dataset path
    :param flag: face detect flag
    :param model_name: select preprocessing method
    :param return_dict: : preprocessed image, label

    """

    preprocessed_label = label_preprocess(preprocess_type=preprocess_type,
                                          path=path + ground_truth_name,
                                          **kwargs)

    rst_dict = video_preprocess(preprocess_type=preprocess_type,
                                path=path + vid_name,
                                **kwargs)
    if None in rst_dict :
        return

    # ppg, sbp, dbp, hr
    # if preprocess_type in ["DeepPhys", "PhysNet", "PhysNet_LSTM"]:
    return_dict[path.replace('/', '') + str(kwargs['flip_flag'])] = {
        'preprocessed_video': rst_dict["video_data"],
        'frame_number': rst_dict["frame_number"],
        'flip_arr': rst_dict["flip_arr"],
        'keypoint' : rst_dict["keypoint"],
        'raw_video' : rst_dict["raw_video"],
        'preprocessed_label': preprocessed_label}
    # elif preprocess_type in ["TEST"]:
    #     return_dict[path.replace('/', '')] = {
    #         'keypoint': rst_dict["keypoint"],
    #         'raw_video': rst_dict["raw_video"],
    #         'preprocessed_label': preprocessed_label }
        # 'preprocessed_hr': preprocessed_hr}


def chunk_preprocessing(preprocess_type,
                        data_list,
                        dataset_root_path,
                        vid_name,
                        ground_truth_name,
                        dataset_name,
                        dataset_path,
                        face_detect_algorithm,
                        fixed_position,
                        img_size,
                        chunk_size,
                        idx):


    process = []
    save_root_path = dataset_path

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    for index, data_path in enumerate(data_list):
        # for i in range(chunk_size):
        proc = multiprocessing.Process(target=preprocess_Dataset,
                                       args=(
                                           preprocess_type, dataset_root_path + "/" + data_path, vid_name,
                                           ground_truth_name,
                                           return_dict)
                                       , kwargs={"face_detect_algorithm": face_detect_algorithm,
                                                 "fixed_position": fixed_position,
                                                 "img_size": img_size,
                                                 "flip_flag": 0})

        process.append(proc)
        proc.start()
    for proc in process:
        proc.join()


    dataset_path = h5py.File(save_root_path + preprocess_type + "_" + dataset_name + "_" + str(idx) + ".hdf5",
                             "w")

    for index, data_path in enumerate(return_dict.keys()):
        dset = dataset_path.create_group(data_path)
        dset['preprocessed_video'] = return_dict[data_path]['preprocessed_video']
        dset['frame_number'] = return_dict[data_path]['frame_number']
        dset['preprocessed_label'] = return_dict[data_path]['preprocessed_label'][0]
        dset['preprocessed_hr'] = return_dict[data_path]['preprocessed_label'][1]
        dset['flip_arr'] = return_dict[data_path]['flip_arr']
        dset['keypoint'] = return_dict[data_path]['keypoint']
        dset['raw_video'] = return_dict[data_path]['raw_video']
    dataset_path.close()

    manager.shutdown()
