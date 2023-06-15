import multiprocessing
import os

import h5py
import math

from rppg.preprocessing.image_preprocess import video_preprocess
from rppg.preprocessing.text_preprocess import label_preprocess


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
        if dataset['type'] == 'CONT':
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
                        if len(os.listdir(dataset_root_path + tmp)) == 5 and source == 'source1':
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
        elif dataset_name.__contains__("MMPD"):
            data_list = []
            subject_list = [data for data in os.listdir(dataset_root_path) if data.__contains__("subject")]
            for subject in subject_list:
                task_list = [task for task in os.listdir(dataset_root_path + "/" + subject) if task.__contains__('.mat')]
                for task in task_list:
                    tmp = subject + "/" + task
                    data_list.append(tmp)
            vid_name = ""
            ground_truth_name = ""

        ssl_flag = True

        # multiprocessing
        chunk_num = math.ceil(len(data_list) / chunk_size)
        if chunk_num == 1:
            chunk_size = len(data_list)
        for i in range(chunk_num):
            if i == chunk_num - 1:
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

def mkdir_p(directory):
    """Like mkdir -p ."""
    if not directory:
        return
    if directory.endswith("/"):
        mkdir_p(directory[:-1])
        return
    if os.path.isdir(directory):
        return
    mkdir_p(os.path.dirname(directory))
    os.mkdir(directory)


def preprocess_Dataset(preprocess_type, path, vid_name, ground_truth_name, return_dict, **kwargs):
    """
    :param path: dataset path
    :param flag: face detect flag
    :param model_name: select preprocessing method
    :param return_dict: : preprocessed image, label

    """
    save_root_path = kwargs['save_root_path']
    dataset_name = kwargs['dataset_name']


    raw_video = video_preprocess(preprocess_type=preprocess_type,
                                path=path + vid_name,
                                **kwargs)

    preprocessed_label = label_preprocess(preprocess_type=preprocess_type,
                                          path=path + ground_truth_name,
                                          frame_total = len(raw_video),
                                          **kwargs)


    if None in raw_video:
        return

    path = path.split('/')

    add_info = ''

    if dataset_name == "VIPL_HR":
        add_info = path[-3] + "/" + path[-2] + "/"
    if dataset_name == "MMPD":
        add_info = path[-2] + "/"
        path[-1] = path[-1][:-4]

    dir_path = save_root_path + "/" + dataset_name + "/" + preprocess_type + "/" + add_info
    if not os.path.isdir(dir_path):
        mkdir_p(dir_path)

    data = h5py.File(dir_path + path[-1] + ".hdf5","w")
    data.create_dataset('raw_video',data=raw_video)
    data.create_dataset('preprocessed_label', data=preprocessed_label[0])
    data.create_dataset('preprocessed_hr',data=preprocessed_label[1])
    data.close()



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
                                                 "save_root_path": save_root_path,
                                                 "dataset_name" : dataset_name,
                                                 "fixed_position": fixed_position,
                                                 "img_size": img_size,
                                                 "flip_flag": 0})

        process.append(proc)
        proc.start()
    for proc in process:
        proc.join()

    manager.shutdown()
