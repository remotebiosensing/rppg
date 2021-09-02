import multiprocessing
import os

import h5py

from utils.image_preprocess import Deepphys_preprocess_Video, PhysNet_preprocess_Video, RTNet_preprocess_Video
from utils.seq_preprocess import PPNet_preprocess_Mat
from utils.text_preprocess import Deepphys_preprocess_Label, PhysNet_preprocess_Label


def preprocessing(save_root_path: str = "/media/hdd1/dy_dataset/",
                  model_name: str = "DeepPhys",
                  data_root_path: str = "/media/hdd1/",
                  dataset_name: str = "UBFC",
                  train_ratio: float = 0.8):
    """
    :param save_root_path: save file destination path
    :param model_name: select preprocessing method
    :param data_root_path: data set root path
    :param dataset_name: data set name(ex. UBFC, COFACE)
    :param train_ratio: data split [ train ratio : 1 - train ratio]
    :return:
    """
    dataset_root_path = data_root_path + dataset_name

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    if dataset_name == "UBFC":
        data_list = [data for data in os.listdir(dataset_root_path) if data.__contains__("subject11")]
    elif dataset_name == "cuff_less_blood_pressure":
        data_list = [data for data in os.listdir(dataset_root_path) if data.__contains__("part")]

    process = []

    # multiprocessing
    for index, data_path in enumerate(data_list):
        proc = multiprocessing.Process(target=preprocess_Dataset,
                                       args=(dataset_root_path + "/" + data_path, True, model_name, return_dict))
        process.append(proc)
        proc.start()

    for proc in process:
        proc.join()

    train = int(len(return_dict.keys()) * train_ratio)  # split dataset

    train_file = h5py.File(save_root_path + model_name + "_" + dataset_name + "_train.hdf5", "w")

    if model_name in ["DeepPhys", "PhysNet", "PhysNet_LSTM"]:

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


def preprocess_Dataset(path, flag, model_name, return_dict):
    """
    :param path: dataset path
    :param flag: face detect flag
    :param model_name: select preprocessing method
    :param return_dict: : preprocessed image, label
    """

    # Video Based
    if model_name == "DeepPhys":
        rst, preprocessed_video = Deepphys_preprocess_Video(path + "/vid.avi", flag)
    elif model_name == "PhysNet" or model_name == "PhysNet_LSTM":
        rst, preprocessed_video = PhysNet_preprocess_Video(path + "/vid.avi", flag)
    elif model_name == "RTNet":
        rst, preprocessed_video = RTNet_preprocess_Video(path + "/vid.avi", flag)
    elif model_name == "PPNet":    # Sequence data based
        ppg, sbp, dbp, hr  = PPNet_preprocess_Mat(path)


    if model_name in ["DeepPhys","MTTS","PhysNet","PhysNet_LSTM"]:  # can't detect face
        if not rst:
            return

    if model_name == "DeepPhys":
        preprocessed_label = Deepphys_preprocess_Label(path + "/ground_truth.txt")
    elif model_name == "PhysNet" or model_name == "PhysNet_LSTM":
        preprocessed_label = PhysNet_preprocess_Label(path + "/ground_truth.txt")

    # ppg, sbp, dbp, hr
    if model_name in ["DeepPhys", "PhysNet", "PhysNet_LSTM"]:
        return_dict[path.split("/")[-1]] = {'preprocessed_video': preprocessed_video,
                                            'preprocessed_label': preprocessed_label}
    elif model_name in ["PPNet"]:
        return_dict[path.split("/")[-1]] = {'ppg': ppg,'sbp': sbp,'dbp': dbp,'hr' : hr}
