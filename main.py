import multiprocessing
import os
import time
import datetime
import numpy as np
from utils.image_preprocess import preprocess_Video
from utils.text_preprocess import preprocess_Label
import h5py


def dataset_Preprocess(path, flag, return_dict):
    '''
    :param path: dataset path
    :param flag: face detect flag
    :return: preprocessed image, label
    '''
    rst, preprocessed_video = preprocess_Video(path + "/vid.avi", flag)
    if not rst:  # can't detect face
        return
    preprocessed_label = preprocess_Label(path + "/ground_truth.txt")
    return_dict[path.split("/")[-1]] = {'preprocessed_video': preprocessed_video,
                                        'preprocessed_label': preprocessed_label}
    # return preprocessed_video, preprocessed_label


save_root_path = "/media/hdd1/dy_dataset/"
data_root_path = "/media/hdd1/"
dataset_name = "UBFC"
dataset_root_path = data_root_path + dataset_name
result = multiprocessing.Queue()

manager = multiprocessing.Manager()
return_dict = manager.dict()

data_list = [data for data in os.listdir(dataset_root_path) if data.__contains__("subject")]
print(data_list)

procs = []

t = time.time()
# frame 수 체크

for index, data_path in enumerate(data_list):
    proc = multiprocessing.Process(target=dataset_Preprocess,
                                   args=(dataset_root_path + "/" + data_path, True, return_dict))
    procs.append(proc)
    proc.start()

for proc in procs:
    proc.join()

data_file = h5py.File(save_root_path + dataset_name + "_train.hdf5", "w")
for index, data_path in enumerate(return_dict.keys()):
    dset = data_file.create_group(data_path)
    dset['preprocessed_video'] = return_dict[data_path]['preprocessed_video']
    dset['preprocessed_label'] = return_dict[data_path]['preprocessed_label']

data_file.close()

print(datetime.timedelta(seconds=time.time() - t))
