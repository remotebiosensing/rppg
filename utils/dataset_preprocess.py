import multiprocessing
import os

import h5py

from utils.image_preprocess import Deepphys_preprocess_Video, PhysNet_preprocess_Video
from utils.text_preprocess import Deepphys_preprocess_Label, PhysNet_preprocess_Label, cohface_Label


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

    data_list = [data for data in os.listdir(dataset_root_path) if data.__contains__("subject") or data.isdigit()]

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
    for index, data_path in enumerate(return_dict.keys()[:train]):
        dset = train_file.create_group(data_path)
        if model_name =='MetaPhys_task':
            for i in range(len(os.listdir(dataset_root_path + "/" + data_path))):
                ddset = dset.create_group(str(i))
                ddset['preprocessed_video'] = return_dict[data_path]['preprocessed_video'][i]
                ddset['preprocessed_label'] = return_dict[data_path]['preprocessed_label'][i]
        else:
            dset['preprocessed_video'] = return_dict[data_path]['preprocessed_video']
            dset['preprocessed_label'] = return_dict[data_path]['preprocessed_label']
    train_file.close()

    test_file = h5py.File(save_root_path + model_name + "_" + dataset_name + "_test.hdf5", "w")
    for index, data_path in enumerate(return_dict.keys()[train:]):
        dset = test_file.create_group(data_path)
        if model_name == 'MetaPhys_task':
            for i in range(len(os.listdir(dataset_root_path + "/" + data_path))):
                ddset = dset.create_group(str(i))
                ddset['preprocessed_video'] = return_dict[data_path]['preprocessed_video'][i]
                ddset['preprocessed_label'] = return_dict[data_path]['preprocessed_label'][i]
        else:
            dset['preprocessed_video'] = return_dict[data_path]['preprocessed_video']
            dset['preprocessed_label'] = return_dict[data_path]['preprocessed_label']
    test_file.close()


def preprocess_Dataset(path, flag, model_name, return_dict):
    """
    :param path: dataset path
    :param flag: face detect flag
    :param model_name: select preprocessing method
    :param return_dict: : preprocessed image, label
    """
    if model_name == "DeepPhys" or model_name == "MetaPhys":
        rst, preprocessed_video = Deepphys_preprocess_Video(path + "/vid.avi", flag)
    elif model_name == "PhysNet" or model_name == "PhysNet_LSTM":
        rst, preprocessed_video = PhysNet_preprocess_Video(path + "/vid.avi", flag)
    elif model_name == "MetaPhys_task":
        preprocessed_video = []
        preprocessed_label = []
        for i in os.listdir(path):
            rst, video = Deepphys_preprocess_Video(path + '/' + i + '/data.avi', flag=False)
            label = cohface_Label(path + '/' + i + '/data.hdf5', video)
            preprocessed_video.append(video)
            preprocessed_label.append(label)
    if not rst:  # can't detect face
        return

    if model_name == "DeepPhys" or model_name == "MetaPhys":
        preprocessed_label = Deepphys_preprocess_Label(path + "/ground_truth.txt")
    elif model_name == "PhysNet" or model_name == "PhysNet_LSTM":
        preprocessed_label = PhysNet_preprocess_Label(path + "/ground_truth.txt")

    return_dict[path.split("/")[-1]] = {'preprocessed_video': preprocessed_video,
                                        'preprocessed_label': preprocessed_label}