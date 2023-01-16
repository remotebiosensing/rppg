import os

import h5py
import numpy as np
from torch.utils.data import DataLoader

from dataset.AxisNetDataset import AxisNetDataset
from dataset.DeepPhysDataset import DeepPhysDataset
from dataset.GCNDataset import GCNDataset
from dataset.PPNetDataset import PPNetDataset
from dataset.PhysNetDataset import PhysNetDataset
from dataset.RhythmNetDataset import RhythmNetDataset
from dataset.ETArPPGNetDataset import ETArPPGNetDataset
from dataset.VitamonDataset import VitamonDataset

from params import params

def split_data_loader(**kwargs):
    datasets, batch_size, train_shuffle, test_shuffle = kwargs["datasets"], kwargs["batch_size"], kwargs["train_shuffle"], kwargs["test_shuffle"]
    if datasets.__len__() == 3:
        train_loader = DataLoader(datasets[0], batch_size=batch_size, shuffle=train_shuffle)
        validation_loader = DataLoader(datasets[1], batch_size=batch_size, shuffle=test_shuffle)
        test_loader = DataLoader(datasets[2], batch_size=batch_size, shuffle=test_shuffle)
        return [train_loader, validation_loader, test_loader]
    elif datasets.__len__() == 2:
        train_loader = DataLoader(datasets[0], batch_size=batch_size, shuffle=train_shuffle)
        test_loader = DataLoader(datasets[1], batch_size=batch_size, shuffle=test_shuffle)
        return [train_loader, test_loader]


def dataset_loader():
    '''
    :param save_root_path: save file destination path
    :param model_name : model_name
    :param dataset_name: data set name(ex. UBFC, COFACE)
    :param option:[train, test]
    :return: dataset
    '''
    if params.log_flag:
        print("========= dataset_loader() in" + os.path.basename(__file__))

    cnt = 0
    flag = True
    name = params.model
    if params.model == "GCN" or params.model == "GCN_TEST":
        name = "PhysNet"

    if params.model in ["DeepPhys", "MTTS"]:
        appearance_data = []
        motion_data = []
        target_data = []
    elif params.model in ["PhysNet", "PhysNet_LSTM", "GCN"]:
        video_data = []
        label_data = []
        bpm_data = []
    elif params.model in ["PPNet"]:
        ppg = []
        sbp = []
        dbp = []
        hr = []
    elif params.model in ["RTNet"]:
        face_data = []
        mask_data = []
        target_data = []
    elif params.model in ["AxisNet"]:
        video_data = []
        label_data = []
        ptt_data = []
    elif params.model in ["RhythmNet"]:
        st_map_data = []
        target_data = []
    elif params.model in ["ETArPPGNet"]:
        video_data = []
        label_data = []
    elif params.model in ["Vitamon","Vitamon_phase2"]:
        video_data = []
        label_data = []


    root_file_path = params.save_root_path + name + "_" + params.dataset_name + "_" + params.dataset_date +"_"
    idx = 0
    while True:
        file_name = root_file_path + str(idx) + ".hdf5"
        if not os.path.isfile(file_name):
            print("Stop at ", idx)
            break
        else:
            print("file size : ", os.path.getsize(file_name) / 1024 / 1024, 'MB')
        idx += 1
        file = h5py.File(file_name, "r")
        h5_tree(file)
        for key in file.keys():
            if params.model in ["DeepPhys", "MTTS"]:
                appearance_data.extend(file[key]['preprocessed_video'][:, :, :, -3:])
                motion_data.extend(file[key]['preprocessed_video'][:, :, :, :3])
                target_data.extend(file[key]['preprocessed_label'])
            elif params.model in ["PhysNet", "PhysNet_LSTM", "GCN"]:
                if len(file[key]['preprocessed_video']) == len(file[key]['preprocessed_label']):
                    video_data.extend(file[key]['preprocessed_video'])
                    label_data.extend(file[key]['preprocessed_label'])
            elif params.model in ["PPNet"]:
                ppg.extend(file[key]['ppg'])
                sbp.extend(file[key]['sbp'])
                dbp.extend(file[key]['dbp'])
                hr.extend(file[key]['hr'])
            elif params.model in ["RTNet"]:
                face_data.extend(file[key]['preprocessed_video'][:, :, :, -3:])
                mask_data.extend(file[key]['preprocessed_video'][:, :, :, :3])
                target_data.extend(file[key]['preprocessed_label'])
            elif params.model in ["AxisNet"]:
                video_data.extend(file[key]['preprocessed_video'])
                ptt_data.extend(file[key]['preprocessed_ptt'])
                label_data.extend(file[key]['preprocessed_label'])
            elif params.model in ["RhythmNet"]:
                st_map_data.extend(file[key]['preprocessed_video'])
                target_data.extend(file[key]['preprocessed_label'])
            elif params.model in ["ETArPPGNet"]:
                video_data.extend(file[key]['preprocessed_video'])
                label_data.extend(file[key]['preprocessed_label'])
            elif params.model in ["Vitamon", "Vitamon_phase2"]:
                video_data.extend(file[key]['preprocessed_video'])
                label_data.extend(file[key]['preprocessed_label'])
        file.close()

    if params.model in ["DeepPhys", "MTTS"]:
        dataset = DeepPhysDataset(appearance_data=np.asarray(appearance_data),
                                  motion_data=np.asarray(motion_data),
                                  target=np.asarray(target_data))
    elif params.model in ["PhysNet", "PhysNet_LSTM", "GCN"]:
        if params.model in ["GCN"]:
            dataset = GCNDataset(video_data=np.asarray(video_data),
                                 label_data=np.asarray(label_data),
                                 bpm_data=np.asarray(bpm_data)
                                 )
        elif params.model in ["AxisNet"]:
            dataset = AxisNetDataset(video_data=np.asarray(video_data),
                                     label_data=np.asarray(label_data))
        else:
            dataset = PhysNetDataset(video_data=np.asarray(video_data),
                                     label_data=np.asarray(label_data))
    elif params.model in ["PPNet"]:
        dataset = PPNetDataset(ppg=np.asarray(ppg),
                               sbp=np.asarray(sbp),
                               dbp=np.asarray(dbp),
                               hr=np.asarray(hr))
    elif params.model in ["RTNet"]:
        dataset = PPNetDataset(face_data=np.asarray(face_data),
                               mask_data=np.asarray(mask_data),
                               target=np.asarray(target_data))
    elif params.model in ["AxisNet"]:
        std_shape = (320, 472, 3)  # ptt_data[0].shape
        for i in range(len(ptt_data)):
            if ptt_data[i].shape != std_shape:
                ptt_data[i] = np.resize(ptt_data[i], std_shape)
        dataset = AxisNetDataset(video_data=np.asarray(video_data),
                                 ptt_data=np.asarray(ptt_data),
                                 label_data=np.asarray(label_data), )
    elif params.model in ["RhythmNet"]:
        dataset = RhythmNetDataset(st_map_data=np.asarray(st_map_data),
                                   target_data=np.asarray(target_data))
    elif params.model in ["ETArPPGNet"]:
        dataset = ETArPPGNetDataset(video_data=np.asarray(video_data),
                                    label_data=np.asarray(label_data))

    elif params.model in ["Vitamon","Vitamon_phase2"]:
        dataset = VitamonDataset(video_data=np.asarray(video_data),
                                    label_data=np.asarray(label_data))


    return dataset

def h5_tree(val, pre=''):
    items = len(val)
    for key, val in val.items():
        items -= 1
        if items == 0:
            # the last item
            if type(val) == h5py._hl.group.Group:
                print(pre + '└── ' + key)
                h5_tree(val, pre+'    ')
            else:
                print(pre + '└── ' + key + ' (%d)' % len(val))
        else:
            if type(val) == h5py._hl.group.Group:
                print(pre + '├── ' + key)
                h5_tree(val, pre+'│   ')
            else:
                print(pre + '├── ' + key + ' (%d)' % len(val))