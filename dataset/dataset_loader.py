import h5py
import numpy as np

from dataset.DeepPhysDataset import DeepPhysDataset
from dataset.PPNetDataset import PPNetDataset
from dataset.PhysNetDataset import PhysNetDataset
from dataset.GCNDataset import GCNDataset
from utils.funcs import load_list_of_dicts
def dataset_loader(save_root_path: str = "/media/hdd1/dy_dataset/",
                   model_name: str = "DeepPhys",
                   dataset_name: str = "UBFC",
                   option: str = "train"):
    '''
    :param save_root_path: save file destination path
    :param model_name : model_name
    :param dataset_name: data set name(ex. UBFC, COFACE)
    :param option:[train, test]
    :return: dataset
    '''
    hpy_file = h5py.File(save_root_path + model_name + "_" + dataset_name + "_" + option + ".hdf5", "r")
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
    elif model_name in ["PhysNet", "PhysNet_LSTM"]:
        video_data = []
        label_data = []
        for key in hpy_file.keys():
            video_data.extend(hpy_file[key]['preprocessed_video'])
            label_data.extend(hpy_file[key]['preprocessed_label'])
        hpy_file.close()

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
    elif model_name in ["GCN"]:
        # video_data = []
        label_data = []
        for key in hpy_file.keys():
            # video_data.extend(hpy_file[key]['preprocessed_video'])
            label_data.extend(hpy_file[key]['preprocessed_label'])
        hpy_file.close()
        print("load_list_of_dicts")
        graphs = load_list_of_dicts(graph_file)

        dataset = GCNDataset(
            # video_data=np.asarray(video_data),
                                        graph_data=np.asarray(graphs),
                                         label_data=np.asarray(label_data))

    return dataset
