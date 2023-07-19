import multiprocessing
import os
import dlib
import csv
import json
import h5py
import scipy.io as sio

import cv2
import face_recognition
import math
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from rppg.utils.funcs import detrend, BPF, get_hrv
from tqdm import tqdm
from rppg.utils.data_path import *


def check_preprocessed_data(cfg):
    print("model: ", cfg.fit.model)
    print("fit type: ", cfg.fit.type)
    print("fit image size: ", cfg.fit.img_size)
    print("dataset: ", cfg.fit.train.dataset, cfg.fit.test.dataset)
    # print("preprocess type: ", fit_cfg.preprocess.common.type)
    # print("preprocess image size: ", fit_cfg.preprocess.common.image_size)
    # if fit_cfg.fit.img_size > fit_cfg.preprocess.common.image_size:
    #     print('*** Image size for model input is larger than the preprocessed image *** '
    #           '\n\tPlease check the image size in the config files.')

    if cfg.preprocess.flag:
        if not os.path.exists(cfg.dataset_path + cfg.preprocess.train_dataset.name + "/" + cfg.preprocess.common.type):
            print('Preprocessing train_dataset({}-{}) dataset...'
                  .format(cfg.preprocess.train_dataset.dataset, cfg.preprocess.common.type))
            print("Preprocess type: ", cfg.preprocess.common.type)
            preprocessing(cfg=cfg, dataset=cfg.preprocess.train_dataset)
        else:
            print('Preprocessed {} data already exists.'.format(cfg.preprocess.train_dataset.name))

        if not os.path.exists(cfg.dataset_path + cfg.preprocess.test_dataset.name + "/" + cfg.preprocess.common.type):
            print('Preprocessing test_dataset({}-{}) dataset...'
                  .format(cfg.preprocess.train_dataset.dataset, cfg.preprocess.common.type))
            preprocessing(cfg=cfg, dataset=cfg.preprocess.test_dataset)
        else:
            print('Preprocessed {} data already exists.'.format(cfg.preprocess.test_dataset.name))

    else:
        if not os.path.exists(cfg.dataset_path + cfg.fit.train.dataset + "/" + cfg.fit.type.upper()):
            print('Preprocessing train({}-{}) dataset...'.format(cfg.fit.train.dataset, cfg.fit.type))
            print("Preprocess type: ", cfg.preprocess.common.type)
            if cfg.preprocess.common.type != cfg.fit.type:
                cfg.preprocess.common.type = cfg.fit.type
                # raise ValueError("dataset type in fit_cfg.preprocess and fit_cfg.fit are different")
            print("Preprocess train_dataset name: ", cfg.preprocess.train_dataset.name)
            if cfg.preprocess.train_dataset.name != cfg.fit.train.dataset:
                cfg.preprocess.train_dataset.name = cfg.fit.train.dataset
                # raise ValueError("train_dataset name in fit_cfg.preprocess and fit_cfg.fit are different")
            if cfg.fit.img_size > cfg.preprocess.common.image_size:
                cfg.preprocess.common.image_size = cfg.fit.img_size
                # print('*** Image size for model input is larger than the preprocessed image *** '
                #       '\n\tPlease check the image size in the config files.')
            preprocessing(cfg=cfg, dataset=cfg.preprocess.train_dataset)
        else:
            print('Preprocessed {} data already exists.'.format(cfg.fit.train.dataset))

        if not os.path.exists(cfg.dataset_path + cfg.fit.test.dataset + "/" + cfg.fit.type.upper()):
            print('Preprocessing test({}-{}) dataset...'.format(cfg.fit.test.dataset, cfg.fit.type))
            print("Preprocess type: ", cfg.preprocess.common.type)
            if cfg.preprocess.common.type != cfg.fit.type:
                cfg.preprocess.common.type = cfg.fit.type
                # raise ValueError("dataset type in fit_cfg.preprocess and fit_cfg.fit are different")
            print("Preprocess test_dataset name: ", cfg.preprocess.test_dataset.name)
            if cfg.preprocess.test_dataset.name != cfg.fit.test.dataset:
                cfg.preprocess.test_dataset.name = cfg.fit.test.dataset
                # raise ValueError("test_dataset name in fit_cfg.preprocess and fit_cfg.fit are different")
            if cfg.fit.img_size > cfg.preprocess.common.image_size:
                cfg.preprocess.common.image_size = cfg.fit.img_size
                # print('*** Image size for model input is larger than the preprocessed image *** '
                #       '\n\tPlease check the image size in the config files.')
            preprocessing(cfg=cfg, dataset=cfg.preprocess.test_dataset)
        else:
            print('Preprocessed {} data already exists.'.format(cfg.fit.test.dataset))


def preprocessing(cfg, dataset):
    # def preprocessing(data_root_path, preprocess_cfg, dataset_path):
    """
    :param save_root_path: save file destination path
    :param model_name: select preprocessing method
    :param data_root_path: data set root path
    :param dataset_name: data set name(ex. UBFC, COFACE)
    :return:
    """

    chunk_size = cfg.preprocess.common.process_num
    manager = multiprocessing.Manager()

    if cfg.preprocess.common.type.upper() == 'CONT':
        preprocess_type = 'CONT'
    elif cfg.preprocess.common.type.upper() == 'DIFF':
        preprocess_type = 'DIFF'
    else:
        preprocess_type = 'CUSTOM'

    img_size = cfg.preprocess.common.image_size
    large_box_coef = cfg.preprocess.common.larger_box_coef

    if not os.path.isdir(cfg.data_root_path + dataset.name):
        # os.makedirs(dataset_root_path)
        raise ValueError("dataset path does not exist, check data_root_path in config.yaml")
    return_dict = manager.dict()

    RawDataPathLoader = None
    if dataset.name == "V4V":
        RawDataPathLoader = V4V_RawDataPathLoader(cfg.data_root_path,
                                                  dataset.select_data.flag,
                                                  dataset.select_data.person_list,
                                                  dataset.select_data.task_list)
    elif dataset.name == "UBFC":
        RawDataPathLoader = UBFC_RawDataPathLoader(cfg.data_root_path,
                                                   dataset.select_data.flag,
                                                   dataset.select_data.person_list)
    elif dataset.name == "VIPL_HR":
        RawDataPathLoader = VIPL_HR_RawDataPathLoader(cfg.data_root_path,
                                                      dataset.select_data.flag,
                                                      dataset.select_data.person_list,
                                                      dataset.select_data.task_list,
                                                      dataset.select_data.source_list)
    elif dataset.name == "PURE":
        RawDataPathLoader = PURE_RawDataPathLoader(cfg.data_root_path,
                                                   dataset.select_data.flag,
                                                   dataset.select_data.person_list,
                                                   dataset.select_data.task_list)
    elif dataset.name == "MMPD":
        RawDataPathLoader = MMPD_RawDataPathLoader(cfg.data_root_path,
                                                   dataset.select_data.flag,
                                                   dataset.select_data.person_list,
                                                   dataset.select_data.task_list)
    elif dataset.name == "UBFC_Phys":
        RawDataPathLoader = UBFC_Phys_RawDataPathLoader(cfg.data_root_path,
                                                        dataset.select_data.flag,
                                                        dataset.select_data.person_list,
                                                        dataset.select_data.task_list)
    elif dataset.name == "RLAP":
        RawDataPathLoader = RLAP_RawDataPathLoader(cfg.data_root_path,
                                                   dataset.select_data.flag,
                                                   dataset.select_data.person_list,
                                                   dataset.select_data.task_list)
    elif dataset.name.__contains__("cohface"):
        RawDataPathLoader = COHFACE_RawDataPathLoader(cfg.data_root_path,
                                                      dataset.select_data.flag,
                                                      dataset.select_data.task_list)

    if dataset.name == "cuff_less_blood_pressure":
        dataset_root_path = cfg.data_root_path + dataset.name
        data_list = [data for data in os.listdir(dataset_root_path) if data.__contains__("part")]
        vid_name = ''
        ground_truth_name = ''
    else:
        dataset_root_path = RawDataPathLoader.dataset_root_path
        data_list = RawDataPathLoader.data_list
        vid_name = RawDataPathLoader.video_name
        ground_truth_name = RawDataPathLoader.ppg_name

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
                            dataset.name, cfg.dataset_path, img_size=img_size, large_box_coef=large_box_coef)


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


def preprocess_Dataset(preprocess_type, dataset_root_path, data_path, vid_name, ground_truth_name, return_dict,
                       **kwargs):
    """
    :param path: dataset path
    :param flag: face detect flag
    :param model_name: select preprocessing method
    :param return_dict: : preprocessed image, label
    """

    save_root_path = kwargs['save_root_path']
    dataset_name = kwargs['dataset_name']
    img_size = kwargs['img_size']

    if dataset_name == "UBFC_Phys":
        data_path = data_path.split('/')
        video_path = dataset_root_path + '/' + data_path[-2] + '/' + 'vid_' + data_path[-1] + vid_name
        label_path = dataset_root_path + '/' + data_path[-2] + '/' + 'bvp_' + data_path[-1] + ground_truth_name
    else:
        video_path = dataset_root_path + data_path + vid_name
        label_path = dataset_root_path + data_path + ground_truth_name

    raw_video, preprocessed_label, hrv = data_preprocess(preprocess_type, video_path, label_path, **kwargs)
    # raw_video, preprocessed_label, hrv = [1], 2, 3  # For Debug

    if None in raw_video:
        return

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

    dir_path = save_root_path + "/" + dataset_name + "/" + preprocess_type + "/" + add_info
    if not os.path.isdir(dir_path):
        mkdir_p(dir_path)

    data = h5py.File(dir_path + data_path + ".hdf5", "w")
    data.create_dataset('raw_video', data=raw_video)
    data.create_dataset('preprocessed_label', data=preprocessed_label)
    data.create_dataset('hrv', data=hrv)
    data.close()


def chunk_preprocessing(preprocess_type, data_list, dataset_root_path, vid_name, ground_truth_name, dataset_name,
                        dataset_path, img_size, large_box_coef):
    process = []
    save_root_path = dataset_path

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    for index, data_path in enumerate(data_list):
        proc = multiprocessing.Process(target=preprocess_Dataset,
                                       args=(preprocess_type, dataset_root_path, data_path, vid_name,
                                             ground_truth_name, return_dict)
                                       , kwargs={"save_root_path": save_root_path,
                                                 "dataset_name": dataset_name,
                                                 "img_size": img_size,
                                                 "large_box_coef": large_box_coef})

        process.append(proc)
        proc.start()
    for proc in process:
        proc.join()

    manager.shutdown()


def data_preprocess(preprocess_type, video_path, label_path, **kwargs):
    img_size = kwargs['img_size']
    large_box_coef = kwargs['large_box_coef']
    # detection_model = 'cnn' if dlib.DLIB_USE_CUDA else 'hog'
    detection_model = 'hog'
    xy_points = pd.DataFrame(columns=['bottom', 'right', 'top', 'left'])

    # for PURE dataset
    if video_path.__contains__("png"):
        path = video_path[:-4]
        data = sorted(os.listdir(path))[1:]
        frame_total = len(data)
        raw_label = get_label(label_path, frame_total)
        hrv = get_hrv_label(raw_label, fs=30.)

        for i in tqdm(range(frame_total), position=0, leave=True, desc=path):
            frame = cv2.imread(path + "/" + data[i])
            face_locations = face_recognition.face_locations(frame, 1, model=detection_model)
            if len(face_locations) >= 1:
                xy_points.loc[i] = face_locations[0]
            else:
                xy_points.loc[i] = (np.NaN, np.NaN, np.NaN, np.NaN)

        valid_fr_idx = xy_points[xy_points['top'].notnull()].index.tolist()
        front_idx = valid_fr_idx[0]
        rear_idx = valid_fr_idx[-1]

        xy_points = xy_points[front_idx:rear_idx + 1]
        raw_label = raw_label[front_idx:rear_idx + 1]
        hrv = hrv[front_idx:rear_idx + 1]

        y_x_w = get_CntYX_Width(xy_points=xy_points, large_box_coef=large_box_coef)

        raw_video = np.empty((xy_points.__len__(), img_size, img_size, 3))
        for i, frame_num in enumerate(range(front_idx, rear_idx + 1)):
            frame = cv2.imread(path + "/" + data[frame_num])
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = np.take(frame, range(y_x_w[i][0] - y_x_w[i][2],
                                        y_x_w[i][0] + y_x_w[i][2]), 0, mode='clip')
            face = np.take(face, range(y_x_w[i][1] - y_x_w[i][2],
                                       y_x_w[i][1] + y_x_w[i][2]), 1, mode='clip')
            face = (face / 255.).astype(np.float32)
            if img_size == y_x_w[frame_num][2] * 2:
                raw_video[i] = face
            else:
                raw_video[i] = cv2.resize(face, (img_size, img_size), interpolation=cv2.INTER_AREA)

    elif video_path.__contains__(".mat"):
        f = sio.loadmat(video_path)
        frames = f['video']
        raw_label = f['GT_ppg']
        frame_total = len(frames)
        hrv = get_hrv_label(raw_label, fs=30.)

        for i in tqdm(range(frame_total), position=0, leave=True, desc=video_path):
            frame = frames[i]
            face_locations = face_recognition.face_locations((frame * 255.).astype(np.uint8), 1, model=detection_model)
            if len(face_locations) >= 1:
                xy_points.loc[i] = face_locations[0]
            else:
                xy_points.loc[i] = (np.NaN, np.NaN, np.NaN, np.NaN)

        valid_fr_idx = xy_points[xy_points['top'].notnull()].index.tolist()
        front_idx = valid_fr_idx[0]
        rear_idx = valid_fr_idx[-1]

        xy_points = xy_points[front_idx:rear_idx + 1]
        raw_label = raw_label[front_idx:rear_idx + 1]
        hrv = hrv[front_idx:rear_idx + 1]
        frames = frames[front_idx:rear_idx + 1]

        y_x_w = get_CntYX_Width(xy_points=xy_points, large_box_coef=large_box_coef)

        raw_video = np.empty((xy_points.__len__(), img_size, img_size, 3), dtype=np.float32)
        for frame_num, frame in enumerate(frames):
            face = np.take(frame, range(y_x_w[frame_num][0] - y_x_w[frame_num][2],
                                        y_x_w[frame_num][0] + y_x_w[frame_num][2]), 0, mode='clip')
            face = np.take(face, range(y_x_w[frame_num][1] - y_x_w[frame_num][2],
                                       y_x_w[frame_num][1] + y_x_w[frame_num][2]), 1, mode='clip')
            if img_size == y_x_w[frame_num][2] * 2:
                raw_video[frame_num] = face
            else:
                raw_video[frame_num] = cv2.resize(face, (img_size, img_size), interpolation=cv2.INTER_AREA)
    # for UBFC, VIPL-HR dataset
    else:
        cap = cv2.VideoCapture(video_path)
        frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        raw_label = get_label(label_path, frame_total)
        hrv = get_hrv_label(raw_label, fs=30.)

        for i in tqdm(range(frame_total), position=0, leave=True, desc=video_path):
            ret, frame = cap.read()
            if ret:
                face_locations = face_recognition.face_locations(frame, 1, model=detection_model)
                if len(face_locations) >= 1:
                    xy_points.loc[i] = face_locations[0]
                else:
                    xy_points.loc[i] = (np.NaN, np.NaN, np.NaN, np.NaN)
            else:
                break
        cap.release()

        valid_fr_idx = xy_points[xy_points['top'].notnull()].index.tolist()
        front_idx = valid_fr_idx[0]
        rear_idx = valid_fr_idx[-1]

        xy_points = xy_points[front_idx:rear_idx + 1]
        raw_label = raw_label[front_idx:rear_idx + 1]
        hrv = hrv[front_idx:rear_idx + 1]

        y_x_w = get_CntYX_Width(xy_points=xy_points, large_box_coef=large_box_coef)

        cap = cv2.VideoCapture(video_path)
        for _ in range(front_idx):
            cap.read()

        raw_video = np.empty((xy_points.__len__(), img_size, img_size, 3), dtype=np.float32)
        for frame_num in range(rear_idx + 1):
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not ret:
                print(f"Can't receive frame: {video_path}")
                break
            # crop face from frame
            face = np.take(frame, range(y_x_w[frame_num][0] - y_x_w[frame_num][2],
                                        y_x_w[frame_num][0] + y_x_w[frame_num][2]), 0, mode='clip')
            face = np.take(face, range(y_x_w[frame_num][1] - y_x_w[frame_num][2],
                                       y_x_w[frame_num][1] + y_x_w[frame_num][2]), 1, mode='clip')
            face = (face / 255.).astype(np.float32)
            if img_size == y_x_w[frame_num][2] * 2:
                raw_video[frame_num] = face
            else:
                raw_video[frame_num] = cv2.resize(face, (img_size, img_size), interpolation=cv2.INTER_AREA)
        cap.release()
    '''비디오 통째로 고칠거면 여기'''
    if preprocess_type == 'DIFF':
        raw_video = diff_normalize_video(raw_video)
        raw_label = diff_normalize_label(raw_label)
    elif preprocess_type == 'CUSTOM':
        pass

    return raw_video, raw_label, hrv


def get_label(label_path, frame_total):
    # Load input
    if label_path.__contains__("hdf5"):
        f = h5py.File(label_path, 'r')
        label = np.asarray(f['pulse'])

        # label = decimate(label,int(len(label)/frame_total))
        label_bvp = bvp.bvp(label, 256, show=False)
        label = label_bvp['filtered']

        # label = smooth(label, 128)
        label = resample_poly(label, 15, 128)
        # label = resample(label,frame_total)
        # label = detrend(label,100)

        start = label_bvp['onsets'][3]
        end = label_bvp['onsets'][-2]
        label = label[start:end]
        # plt.plot(label)
        # label = resample(label,frame_total)
        label -= np.mean(label)
        label /= np.std(label)
        start = math.ceil(start / 32)
        end = math.floor(end / 32)
    elif label_path.__contains__("json"):
        name = label_path.split("/")
        label = []
        label_time = []
        label_hr = []
        time = []
        with open(label_path[:-4] + name[-2] + ".json") as json_file:
            json_data = json.load(json_file)
            for data in json_data['/FullPackage']:
                label.append(data['Value']['waveform'])
                label_time.append(data['Timestamp'])
                label_hr.append(data['Value']['pulseRate'])
            for data in json_data['/Image']:
                time.append(data['Timestamp'])


    elif label_path.__contains__("csv"):
        f = open(label_path, 'r')
        rdr = csv.reader(f)
        fr = list(rdr)
        label = np.asarray(fr[1:]).reshape((-1)).astype(np.float32)

        # print("label length" + str(len(label)))
        f.close()

        f_time = open(label_path[:-8] + "time.txt", 'r')
        fr_time = f_time.read().split('\n')
        time = np.asarray(fr_time[:-1]).astype(np.float32)  ## 동영상 시간
        f_time.close()

        x = np.linspace(time[0], time[-1], len(label))
        new_x = np.linspace(time[0], time[-1], len(time))
        f = interp1d(x, label)
        label = f(new_x)

        f_hr = open(label_path[:-8] + "gt_HR.csv", 'r')
        rdr = csv.reader(f_hr)
        fr = list(rdr)
        label_hr = np.asarray(fr[1:]).reshape((-1)).astype(np.float32)
        f_hr.close()

        # print("Check")
    elif label_path.__contains__("label.txt"):
        cap = cv2.VideoCapture(label_path[:-9] + "video.mkv")
        frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

        length = int(frame_total / fps * 1000)

        f = open(label_path, 'r')
        f_read = f.read().split('\n')
        f.close()
        label = list(map(float, f_read[:-1]))
        new_label = label[:length:40]
        print("A")
    else:
        f = open(label_path, 'r')
        f_read = f.read().split('\n')
        label = ' '.join(f_read[0].split()).split()
        label_hr = ' '.join(f_read[1].split()).split()
        label = list(map(float, label))
        label = np.array(label).astype('float32')
        label_hr = list(map(float, label_hr))
        label_hr = np.array(label_hr).astype('int')
        f.close()

    label = list(map(float, label))
    if len(label) != frame_total:
        label = np.interp(
            np.linspace(
                1, len(label), frame_total), np.linspace(
                1, len(label), len(label)), label)

    return np.array(label, dtype=np.float32)


def get_hrv_label(ppg_signal, fs=30.):
    clean_ppg = detrend(ppg_signal, 100)
    clean_ppg = BPF(clean_ppg, fs=fs)
    hrv = get_hrv(clean_ppg, fs=fs)
    return hrv.astype(np.float32)


def diff_normalize_label(label):
    delta_label = np.diff(label, axis=0)
    delta_label /= np.std(delta_label)
    delta_label = np.array(delta_label).astype(np.float32)
    delta_label = np.append(delta_label, np.zeros(1, dtype=np.float32), axis=0)
    delta_label[np.isnan(delta_label)] = 0
    return delta_label


def diff_normalize_video(video_data):
    frame_total, h, w, c = video_data.shape

    raw_video = np.empty((frame_total - 1, h, w, 6), dtype=np.float32)
    padd = np.zeros((1, h, w, 6), dtype=np.float32)
    for frame_num in range(frame_total - 1):
        raw_video[frame_num, :, :, :3] = generate_MotionDifference(video_data[frame_num], video_data[frame_num + 1])
    raw_video[:, :, :, :3] = raw_video[:, :, :, :3] / np.std(raw_video[:, :, :, :3])
    raw_video = np.append(raw_video, padd, axis=0)
    video_data = video_data - np.mean(video_data)
    raw_video[:, :, :, 3:] = video_data / np.std(video_data)
    raw_video[:, :, :, 3:] = video_data
    raw_video[np.isnan(raw_video)] = 0
    return raw_video


def generate_MotionDifference(prev_frame, crop_frame):
    '''
    :param prev_frame: previous frame
    :param crop_frame: current frame
    :return: motion diff frame
    '''
    # motion input
    motion_input = (crop_frame - prev_frame) / (crop_frame + prev_frame + 0.000000001)
    # TODO : need to diminish outliers [ clipping ]
    # motion_input = motion_input / np.std(motion_input)
    # TODO : do not divide each D frame, modify divide whole video's unit standard deviation
    return motion_input


def get_CntYX_Width(xy_points, large_box_coef):
    y_range_ext = (xy_points.top - xy_points.bottom) * 0.2  # for forehead
    xy_points.bottom = xy_points.bottom - y_range_ext

    xy_points['cnt_y'] = ((xy_points.top + xy_points.bottom) / 2)
    xy_points['cnt_x'] = ((xy_points.right + xy_points.left) / 2)
    xy_points['bbox_half_size'] = ((xy_points.top - xy_points.bottom).median() * (large_box_coef / 2))
    # TODO: dynamic bbox size (ZoomIn ZoomOut)
    # xy_points['bbox_half_size'] = ((xy_points.top - xy_points.bottom) * (large_box_coef / 2))

    xy_points = xy_points.interpolate()

    xy_points.cnt_x = xy_points.cnt_x.ewm(alpha=0.1).mean()
    xy_points.cnt_y = xy_points.cnt_y.ewm(alpha=0.1).mean()
    xy_points.bbox_half_size = xy_points.bbox_half_size.ewm(alpha=0.1).mean()

    xy_points = xy_points.round().astype(int)
    return xy_points[['cnt_y', 'cnt_x', 'bbox_half_size']].values


def create_debug_video(raw_video, save_path, video_name, fps=30.):
    pathOut = save_path + '/' f'{video_name}.mp4'
    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'mp4v'), fps, (raw_video.shape[2], raw_video.shape[1]))
    for frame_num in range(raw_video.shape[0]):
        frame = (raw_video[frame_num, :, :, :] * 255.).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()

# dataPath = '/home/jh/dataset/PURE/DIFF/01-06.hdf5'
# ubfc_h5 = h5py.File(dataPath, 'r')
# raw_video = np.array(ubfc_h5['raw_video'])
# raw_video = raw_video[:,:,:,3:]
# save_path = '/home/jh/prep_test'
# video_name = 'test_UBFC.mp4'
# create_debug_video(raw_video, save_path, video_name, fps=30.)
# A = raw_video[:,:,:,3:]
