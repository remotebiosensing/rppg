import os

import h5py
import numpy as np
from torch.utils.data import DataLoader

from dataset.AxisNetDataset import AxisNetDataset
from dataset.DeepPhysDataset import DeepPhysDataset
from dataset.GCNDataset import GCNDataset
from dataset.PPNetDataset import PPNetDataset
from dataset.PhysNetDataset import PhysNetDataset


def split_data_loader(datasets, batch_size, train_shuffle, test_shuffle=False):
    if datasets.__len__() == 3:
        train_loader = DataLoader(datasets[0], batch_size=batch_size, shuffle=train_shuffle)
        validation_loader = DataLoader(datasets[1], batch_size=batch_size, shuffle=test_shuffle)
        test_loader = DataLoader(datasets[2], batch_size=batch_size, shuffle=test_shuffle)
        return [train_loader, validation_loader, test_loader]
    elif datasets.__len__() == 2:
        train_loader = DataLoader(datasets[0], batch_size=batch_size, shuffle=train_shuffle)
        test_loader = DataLoader(datasets[1], batch_size=batch_size, shuffle=test_shuffle)
        return [train_loader, test_loader]


def dataset_loader(save_root_path: str = "/media/hdd1/dy/dataset/",
                   model_name: str = "DeepPhys",
                   dataset_name: str = "UBFC",
                   option: str = "train",
                   log_flag: bool = True
                   ):
    '''
    :param save_root_path: save file destination path
    :param model_name : model_name
    :param dataset_name: data set name(ex. UBFC, COFACE)
    :param option:[train, test]
    :return: dataset
    '''
    if log_flag:
        print("========= dataset_loader() in" + os.path.basename(__file__))

    cnt = 0
    flag = True
    name = model_name
    if model_name == "GCN" or model_name == "GCN_TEST":
        name = "PhysNet"

    train_file = save_root_path + name + "_" + dataset_name + "_" + "train" + ".hdf5"
    test_file = save_root_path + name + "_" + dataset_name + "_" + "test" + ".hdf5"
    hpy_train_file = h5py.File(train_file, "r")
    hpy_test_file = h5py.File(test_file, "r")

    print("train file size : ", os.path.getsize(train_file)/1024/1024,'MB')
    print("test file size : ", os.path.getsize(test_file)/1024/1024,'MB')


    graph_file = save_root_path + model_name + "_" + dataset_name + "_" + option + ".pkl"

    if model_name in ["DeepPhys", "MTTS"]:
        appearance_data = []
        motion_data = []
        target_data = []

        for key in hpy_file.keys():
            appearance_data.extend(hpy_file[key]['preprocessed_video'][:, :, :, -3:])
            motion_data.extend(hpy_file[key]['preprocessed_video'][:, :, :, :3])
            target_data.extend(hpy_file[key]['preprocessed_label'])

        hpy_file.close()

        dataset = DeepPhysDataset(appearance_data=np.asarray(appearance_data),
                                  motion_data=np.asarray(motion_data),
                                  target=np.asarray(target_data))
    elif model_name in ["PhysNet", "PhysNet_LSTM", "GCN"]:
        video_data = []
        label_data = []
        bpm_data = []
        cnt = 0
        # for key in hpy_train_file.keys():
        #     cnt += 1
        #     if len(hpy_train_file[key]['preprocessed_video']) == len(hpy_train_file[key]['preprocessed_label']):
        #         video_data.extend(hpy_train_file[key]['preprocessed_video'])
        #         label_data.extend(hpy_train_file[key]['preprocessed_label'])
        # bpm_data.extend(hpy_train_file[key]['preprocessed_hr'])

        # if option == "test" or flag :
        # if cnt == 4:
        # break
        # hpy_train_file.close()
        for key in hpy_test_file.keys():
            cnt += 1
            # if cnt <cflag:
            #     continue
            # if cnt < cflag:
            #     continue
            # if cnt > cflag:
            #     break
            if len(hpy_test_file[key]['preprocessed_video']) == len(hpy_test_file[key]['preprocessed_label']):
                video_data.extend(hpy_test_file[key]['preprocessed_video'])
                label_data.extend(hpy_test_file[key]['preprocessed_label'])
                # bpm_data.extend(hpy_test_file[key]['preprocessed_hr'])
            if cnt == 4:
                break
            # if cnt == cflag:
            #     break

            # if option == "test" or flag :
            #     break
        hpy_test_file.close()

        if model_name in ["GCN"]:
            dataset = GCNDataset(video_data=np.asarray(video_data),
                                 label_data=np.asarray(label_data),
                                 bpm_data=np.asarray(bpm_data)
                                 )
        elif model_name in ["AxisNet"]:
            dataset = AxisNetDataset(video_data=np.asarray(video_data),
                                     label_data=np.asarray(label_data))

        else:
            dataset = PhysNetDataset(video_data=np.asarray(video_data),
                                     label_data=np.asarray(label_data))

    elif model_name in ["PPNet"]:
        ppg = []
        sbp = []
        dbp = []
        hr = []

        for key in hpy_file.keys():
            ppg.extend(hpy_file[key]['ppg'])
            sbp.extend(hpy_file[key]['sbp'])
            dbp.extend(hpy_file[key]['dbp'])
            hr.extend(hpy_file[key]['hr'])
        hpy_file.close()

        dataset = PPNetDataset(ppg=np.asarray(ppg),
                               sbp=np.asarray(sbp),
                               dbp=np.asarray(dbp),
                               hr=np.asarray(hr))

    elif model_name in ["RTNet"]:
        face_data = []
        mask_data = []
        target_data = []

        for key in hpy_file.keys():
            face_data.extend(hpy_file[key]['preprocessed_video'][:, :, :, -3:])
            mask_data.extend(hpy_file[key]['preprocessed_video'][:, :, :, :3])
            target_data.extend(hpy_file[key]['preprocessed_label'])
        hpy_file.close()

        dataset = PPNetDataset(face_data=np.asarray(face_data),
                               mask_data=np.asarray(mask_data),
                               target=np.asarray(target_data))
    elif model_name in ["AxisNet"]:
        video_data = []
        label_data = []
        ptt_data = []

        for key in hpy_file.keys():
            video_data.extend(hpy_file[key]['preprocessed_video'])
            ptt_data.extend(hpy_file[key]['preprocessed_ptt'])
            label_data.extend(hpy_file[key]['preprocessed_label'])
        hpy_file.close()

        std_shape = (320, 472, 3)  # ptt_data[0].shape
        for i in range(len(ptt_data)):
            if ptt_data[i].shape != std_shape:
                ptt_data[i] = np.resize(ptt_data[i], std_shape)

        dataset = AxisNetDataset(video_data=np.asarray(video_data),
                                 ptt_data=np.asarray(ptt_data),
                                 label_data=np.asarray(label_data), )

    return dataset
