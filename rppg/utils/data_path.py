
import os
from itertools import product


class UBFC_RawDataPathLoader:
    def __init__(self, data_root_path, select_data_flag=False, person_list=None):
        if person_list is None:
            person_list = []

        self.data_root_path = data_root_path
        self.dataset_root_path = self.data_root_path + 'UBFC'

        # For Preprocessing
        self.video_name = '/vid.avi'
        self.ppg_name = '/ground_truth.txt'
        self.video_fps = 30.
        self.ppg_sample_rate = 30.

        self.person_list = self.set_person_list(person_list, select_data_flag)
        self.data_list = self.set_data_list(select_data_flag)

    def set_person_list(self, person_list, select_data_flag=True):
        if select_data_flag and len(person_list) > 0:
            self.person_list = person_list
        else:
            self.person_list = sorted([int(data[7:]) for data
                                       in os.listdir(self.dataset_root_path) if data.__contains__("subject")])
        return self.person_list

    def set_data_list(self, select_data_flag=True):
        self.data_list = []
        for p in self.person_list:
            data_temp = f'/subject{p}'
            if os.path.isdir(self.dataset_root_path + data_temp):
                self.data_list.append(data_temp)
            elif select_data_flag:
                print(f'Not Fetched Data Path: {data_temp}')
        return self.data_list


class UBFC_Phys_RawDataPathLoader:
    def __init__(self, data_root_path, select_data_flag=False, person_list=None, task_list=None):
        if person_list is None:
            person_list = []
        if task_list is None:
            task_list = []

        self.data_root_path = data_root_path
        self.dataset_root_path = self.data_root_path + 'UBFC_Phys'

        # For Preprocessing
        self.video_name = '.avi'
        self.ppg_name = '.csv'
        self.video_fps = 35.14
        self.ppg_sample_rate = 64.

        self.person_list = self.set_person_list(person_list, select_data_flag)
        self.task_list = self.set_task_list(task_list, select_data_flag)
        self.data_list = self.set_data_list(select_data_flag)

    def set_person_list(self, person_list, select_data_flag=True):
        if select_data_flag and len(person_list) > 0:
            self.person_list = person_list
        else:
            self.person_list = sorted([int(data[1:])
                                       for data in os.listdir(self.dataset_root_path) if data.__contains__("s")])
        return self.person_list

    def set_task_list(self, task_list, select_data_flag=True):
        if select_data_flag and len(task_list) > 0:
            self.task_list = task_list
        else:
            self.task_list = [1, 2, 3]
        return self.task_list

    def set_data_list(self, select_data_flag=True):
        self.data_list = []
        for (p, t) in product(self.person_list, self.task_list):
            data_temp = f'/s{p}/s{p}_T{t}'
            if os.path.isfile(self.dataset_root_path + f'/s{p}/vid_s{p}_T{t}.avi') \
                    and os.path.isfile(self.dataset_root_path + f'/s{p}/bvp_s{p}_T{t}.csv'):
                self.data_list.append(data_temp)
            elif select_data_flag:
                print(f'Not Fetched Data Path: {data_temp}')
        return self.data_list


class PURE_RawDataPathLoader:
    def __init__(self, data_root_path, select_data_flag=False, person_list=None, task_list=None):
        if person_list is None:
            person_list = []
        if task_list is None:
            task_list = []

        self.data_root_path = data_root_path
        self.dataset_root_path = self.data_root_path + 'PURE'

        # For Preprocessing
        self.video_name = '/png'
        self.ppg_name = '/json'
        self.video_fps = 35.14
        self.ppg_sample_rate = 60.

        self.person_list = self.set_person_list(person_list, select_data_flag)
        self.task_list = self.set_task_list(task_list, select_data_flag)
        self.data_list = self.set_data_list(select_data_flag)

    def set_person_list(self, person_list, select_data_flag=True):
        if select_data_flag and len(person_list) > 0:
            self.person_list = person_list
        else:
            self.person_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        return self.person_list

    def set_task_list(self, task_list, select_data_flag=True):
        if select_data_flag and len(task_list) > 0:
            self.task_list = task_list
        else:
            self.task_list = [1, 2, 3, 4, 5, 6]
        return self.task_list

    def set_data_list(self, select_data_flag=True):
        self.data_list = []
        for (p, t) in product(self.person_list, self.task_list):
            data_temp = f'/{p:02}-{t:02}'
            if os.path.isdir(self.dataset_root_path + data_temp):
                self.data_list.append(data_temp)
            elif select_data_flag:
                print(f'Not Fetched Data Path: {data_temp}')
        return self.data_list


class MMPD_RawDataPathLoader:
    def __init__(self, data_root_path, select_data_flag=False, person_list=None, task_list=None):
        if person_list is None:
            person_list = []
        if task_list is None:
            task_list = []

        self.data_root_path = data_root_path
        self.dataset_root_path = self.data_root_path + 'MMPD'

        # For Preprocessing
        self.video_name = '.mat'
        self.ppg_name = '.mat'
        self.video_fps = 30.
        self.ppg_sample_rate = 30.

        self.person_list = self.set_person_list(person_list, select_data_flag)
        self.task_list = self.set_task_list(task_list, select_data_flag)
        self.data_list = self.set_data_list(select_data_flag)

        if select_data_flag and len(person_list) > 0:
            self.person_list = person_list
        else:
            self.person_list = sorted([int(data[7:])
                                       for data in os.listdir(self.dataset_root_path) if data.__contains__("subject")])

        if select_data_flag and len(task_list) > 0:
            self.task_list = task_list
        else:
            self.task_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

        self.data_list = []
        for (p, t) in product(self.person_list, self.task_list):
            data_temp = f'/subject{p}/p{p}_{t}'
            if os.path.isfile(self.dataset_root_path + data_temp + '.mat'):
                self.data_list.append(data_temp)
            elif select_data_flag:
                print(f'Not Fetched Data Path: {data_temp}')

    def set_person_list(self, person_list, select_data_flag=True):
        if select_data_flag and len(person_list) > 0:
            self.person_list = person_list
        else:
            self.person_list = sorted([int(data[7:])
                                       for data in os.listdir(self.dataset_root_path) if data.__contains__("subject")])
        return self.person_list

    def set_task_list(self, task_list, select_data_flag=True):
        if select_data_flag and len(task_list) > 0:
            self.task_list = task_list
        else:
            self.task_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        return self.task_list

    def set_data_list(self, select_data_flag=True):
        self.data_list = []
        for (p, t) in product(self.person_list, self.task_list):
            data_temp = f'/subject{p}/p{p}_{t}'
            if os.path.isfile(self.dataset_root_path + data_temp + '.mat'):
                self.data_list.append(data_temp)
            elif select_data_flag:
                print(f'Not Fetched Data Path: {data_temp}')
        return self.data_list


class V4V_RawDataPathLoader:
    def __init__(self, data_root_path, select_data_flag=False, person_list=None, task_list=None):
        if person_list is None:
            person_list = []
        if task_list is None:
            task_list = []

        self.data_root_path = data_root_path
        self.dataset_root_path = self.data_root_path + 'V4V/train_val/data'

        # For Preprocessing
        self.video_name = '/video.mkv'
        self.ppg_name = '/label.txt'
        self.video_fps = 25.
        self.ppg_sample_rate = 1000.

        self.person_list = self.set_person_list(person_list, select_data_flag)
        self.task_list = self.set_task_list(task_list, select_data_flag)
        self.data_list = self.set_data_list(select_data_flag)

    def set_person_list(self, person_list, select_data_flag=True):
        if select_data_flag and len(person_list) > 0:
            self.person_list = person_list
        else:
            self.person_list = sorted(list(set([data[:4] for data in os.listdir(self.dataset_root_path)])))
        return self.person_list

    def set_task_list(self, task_list, select_data_flag=True):
        if select_data_flag and len(task_list) > 0:
            self.task_list = task_list
        else:
            self.task_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        return self.task_list

    def set_data_list(self, select_data_flag=True):
        self.data_list = []
        for (p, t) in product(self.person_list, self.task_list):
            data_temp = f'/{p}_T{t}'
            if os.path.isdir(self.dataset_root_path + data_temp):
                self.data_list.append(data_temp)
            elif select_data_flag:
                print(f'Not Fetched Data Path: {data_temp}')
        return self.data_list


class VIPL_HR_RawDataPathLoader:
    def __init__(self, data_root_path, select_data_flag=False, person_list=None, task_list=None, source_list=None):
        if person_list is None:
            person_list = []
        if task_list is None:
            task_list = []
        if source_list is None:
            source_list = []

        self.data_root_path = data_root_path
        self.dataset_root_path = self.data_root_path + 'VIPL_HR/data'

        # For Preprocessing
        self.video_name = '/video.avi'
        self.ppg_name = '/wave.csv'
        # self.source_1_video_fps = 25.
        # self.source_2_video_fps = 30.
        # self.source_3_video_fps = 30.
        # self.source_3_video_fps = 30.
        # self.ppg_sample_rate = 0.

        self.person_list = self.set_person_list(person_list, select_data_flag)
        self.task_list = self.set_task_list(task_list, select_data_flag)
        self.source_list = self.set_source_list(source_list, select_data_flag)
        self.data_list = self.set_data_list(select_data_flag)

    def set_person_list(self, person_list, select_data_flag=True):
        if select_data_flag and len(person_list) > 0:
            self.person_list = person_list
        else:
            self.person_list = sorted([int(data[1:]) for data
                                       in os.listdir(self.dataset_root_path) if data.__contains__("p")])
        return self.person_list

    def set_task_list(self, task_list, select_data_flag=True):
        if select_data_flag and len(task_list) > 0:
            self.task_list = task_list
        else:
            self.task_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        return self.task_list

    def set_source_list(self, source_list, select_data_flag=True):
        if select_data_flag and len(source_list) > 0:
            self.source_list = source_list
        else:
            self.source_list = [1, 2, 3, 4]
        return self.source_list

    def set_data_list(self, select_data_flag=True):
        self.data_list = []
        for (p, t, s) in product(self.person_list, self.task_list, self.source_list):
            data_temp = f'/p{p}/v{t}/source{s}'
            if os.path.isdir(self.dataset_root_path + data_temp) and len(
                    os.listdir(self.dataset_root_path + data_temp)) == 5:
                self.data_list.append(data_temp)
            elif select_data_flag:
                print(f'Not Fetched Data Path: {data_temp}')
        return self.data_list


class COHFACE_RawDataPathLoader:
    def __init__(self, data_root_path, select_data_flag=False, task_list=None):
        if task_list is None:
            task_list = ['all']

        self.data_root_path = data_root_path
        self.dataset_root_path = self.data_root_path + 'cohface'

        self.video_name = 'data.mkv'
        self.ppg_name = 'data.hdf5'

        protocol = dataset_root_path + "/" + "protocols/"
        if 'all' in task_list:
            protocol += "all/all.txt"
        elif 'clean' in task_list:
            protocol += "clean/all.txt"
        elif 'natural' in task_list:
            protocol += "natural/all.txt"
        f = open(protocol, 'r')
        data_list = f.readlines()
        self.data_list = [path.replace("data\n", "") for path in data_list]
        f.close()



if __name__ == "__main__":
    from rppg.config import get_config
    preprocess_cfg = get_config("configs/preprocess.yaml")
    dataset_name = 'UBFC_Phys'

    if dataset_name == "V4V":
        RawDataPathLoader = V4V_RawDataPathLoader(preprocess_cfg.data_root_path,
                                                  preprocess_cfg.dataset.select_data.flag,
                                                  preprocess_cfg.dataset.select_data.person_list,
                                                  preprocess_cfg.dataset.select_data.task_list)
    elif dataset_name == "UBFC":
        RawDataPathLoader = UBFC_RawDataPathLoader(preprocess_cfg.data_root_path,
                                                   preprocess_cfg.dataset.select_data.flag,
                                                   preprocess_cfg.dataset.select_data.person_list)
    elif dataset_name == "VIPL_HR":
        RawDataPathLoader = VIPL_HR_RawDataPathLoader(preprocess_cfg.data_root_path,
                                                      preprocess_cfg.dataset.select_data.flag,
                                                      preprocess_cfg.dataset.select_data.person_list,
                                                      preprocess_cfg.dataset.select_data.task_list,
                                                      preprocess_cfg.dataset.select_data.source_list)
    elif dataset_name == "PURE":
        RawDataPathLoader = PURE_RawDataPathLoader(preprocess_cfg.data_root_path,
                                                   preprocess_cfg.dataset.select_data.flag,
                                                   preprocess_cfg.dataset.select_data.person_list,
                                                   preprocess_cfg.dataset.select_data.task_list)
    elif dataset_name == "MMPD":
        RawDataPathLoader = MMPD_RawDataPathLoader(preprocess_cfg.data_root_path,
                                                   preprocess_cfg.dataset.select_data.flag,
                                                   preprocess_cfg.dataset.select_data.person_list,
                                                   preprocess_cfg.dataset.select_data.task_list)
    elif dataset_name == "UBFC_Phys":
        RawDataPathLoader = UBFC_Phys_RawDataPathLoader(preprocess_cfg.data_root_path,
                                                        preprocess_cfg.dataset.select_data.flag,
                                                        preprocess_cfg.dataset.select_data.person_list,
                                                        preprocess_cfg.dataset.select_data.task_list)
    elif dataset_name.__contains__("cohface"):
        RawDataPathLoader = COHFACE_RawDataPathLoader(preprocess_cfg.data_root_path,
                                                      preprocess_cfg.dataset.select_data.flag,
                                                      preprocess_cfg.dataset.select_data.task_list)

    if dataset_name == "cuff_less_blood_pressure":
        dataset_root_path = preprocess_cfg.data_root_path + dataset_name
        data_list = [data for data in os.listdir(dataset_root_path) if data.__contains__("part")]
        vid_name = ''
        ground_truth_name = ''
    else:
        dataset_root_path = RawDataPathLoader.dataset_root_path
        data_list = RawDataPathLoader.data_list
        vid_name = RawDataPathLoader.video_name
        ground_truth_name = RawDataPathLoader.ppg_name


    data_path = data_list[0]
    if dataset_name == "UBFC_Phys":
        data_path = data_path.split('/')
        video_path = dataset_root_path + '/' + data_path[-2] + '/' + 'vid_' + data_path[-1] + vid_name
        label_path = dataset_root_path + '/' + data_path[-2] + '/' + 'bvp_' + data_path[-1] + ground_truth_name
    else:
        video_path = dataset_root_path + data_path + vid_name
        label_path = dataset_root_path + data_path + ground_truth_name

    os.path.isfile(video_path)
    os.path.isfile(label_path)

    add_info = ''

    if dataset_name == "VIPL_HR":
        data_path = data_path.split('/')
        add_info = data_path[-3] + "/" + data_path[-2] + "/"
        data_path = data_path[-1]
    if dataset_name == "MMPD":
        data_path = data_path.split('/')
        add_info = data_path[-2] + "/"
        data_path = data_path[-1]
    if dataset_name == "UBFC_Phys":
        add_info = data_path[-2] + "/"
        data_path = data_path[-1]

    dir_path = '/home/jh/dataset/' + "/" + dataset_name + "/" + 'CONT' + "_" + str(128) + "/" + add_info
    dir_path + data_path + ".hdf5"

    print("END!")