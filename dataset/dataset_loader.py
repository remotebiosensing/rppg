import h5py
import numpy as np
from dataset.DeltaDataset import DeltaDataset


def dataset_loader(save_root_path: str = "/media/hdd1/dy_dataset/",
                   dataset_name: str = "UBFC",
                   option: str = "train"):
    '''
    :param save_root_path: save file destination path
    :param dataset_name: data set name(ex. UBFC, COFACE)
    :param option:[train, test]
    :return: dataset
    '''
    hpy_file = h5py.File(save_root_path + dataset_name + "_"+option+".hdf5", "r")
    appearance_data = []
    motion_data = []
    target_data = []

    for key in hpy_file.keys():
        appearance_data.extend(hpy_file[key]['preprocessed_video'][:, :, :, -3:])
        motion_data.extend(hpy_file[key]['preprocessed_video'][:, :, :, :3])
        target_data.extend(hpy_file[key]['preprocessed_label'])
    hpy_file.close()

    dataset = DeltaDataset(appearance_data=np.asarray(appearance_data),
                           motion_data=np.asarray(motion_data),
                           target=np.asarray(target_data))
    return dataset
