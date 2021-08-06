import multiprocessing
import os

import h5py

from utils.image_preprocess import Deepphys_preprocess_Video, PhysNet_preprocess_Video
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

    data_list = [data for data in os.listdir(dataset_root_path) if data.__contains__("subject")]

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
        dset['preprocessed_video'] = return_dict[data_path]['preprocessed_video']
        dset['preprocessed_label'] = return_dict[data_path]['preprocessed_label']
    train_file.close()

    test_file = h5py.File(save_root_path + model_name + "_" + dataset_name + "_test.hdf5", "w")
    for index, data_path in enumerate(return_dict.keys()[train:]):
        dset = test_file.create_group(data_path)
        dset['preprocessed_video'] = return_dict[data_path]['preprocessed_video']
        dset['preprocessed_label'] = return_dict[data_path]['preprocessed_label']
    test_file.close()


def Meta_preprocessing(save_root_path: str = "/media/hdd1/dy_dataset/",
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

    data_list = [data for data in os.listdir(dataset_root_path) if data.__contains__("subject")]

    process = []

    # multiprocessing
    for index, data_path in enumerate(data_list):
        print('datapreprocessing..',data_path)

        proc = multiprocessing.Process(target=preprocess_Dataset,
                                       args=(dataset_root_path + "/" + data_path, True, model_name, return_dict))
        process.append(proc)
        proc.start()

    for proc in process:
        proc.join()

    for index, data_path in enumerate(return_dict.keys()):
        video = return_dict[data_path]['preprocessed_video']
        label = return_dict[data_path]['preprocessed_label']

        data_len = len(label) //8 # 1개의 데이터를 8개로
        for i in range(8):
            n_video = video[data_len * i:data_len * (i + 1)]
            n_label = label[data_len * i:data_len * (i + 1)]

            dir = save_root_path + model_name + '/'+dataset_name+'/'+data_path
            if not os.path.exists(dir):
                os.makedirs(dir)

            train_file = h5py.File(dir+"/{}_{}.hdf5".format(data_path, i), "w")
            train_file.create_dataset('preprocessed_video', data=n_video)
            train_file.create_dataset('preprocessed_label', data=n_label)
            #print(train_file["preprocessed_video"])
            #print(train_file["preprocessed_label"])
            train_file.close()


def preprocess_Dataset(path, flag, model_name, return_dict):
    """
    :param path: dataset path
    :param flag: face detect flag
    :param model_name: select preprocessing method
    :param return_dict: : preprocessed image, label
    """
    if model_name == "DeepPhys":
        rst, preprocessed_video = Deepphys_preprocess_Video(path + "/vid.avi", flag)
    elif model_name == "PhysNet":
        rst, preprocessed_video = PhysNet_preprocess_Video(path + "/vid.avi", flag)
    elif model_name == "MetaPhys":
        rst, preprocessed_video = Deepphys_preprocess_Video(path + "/vid.avi", flag)
    if not rst:  # can't detect face
        return

    if model_name == "DeepPhys":
        preprocessed_label = Deepphys_preprocess_Label(path + "/ground_truth.txt")
    elif model_name == "PhysNet":
        preprocessed_label = PhysNet_preprocess_Label(path + "/ground_truth.txt")
    elif model_name == "MetaPhys":
        preprocessed_label = Deepphys_preprocess_Label(path + "/ground_truth.txt")

    return_dict[path.split("/")[-1]] = {'preprocessed_video': preprocessed_video,
                                        'preprocessed_label': preprocessed_label}
