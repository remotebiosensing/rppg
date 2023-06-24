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
from tqdm import tqdm


def preprocessing(data_root_path, preprocess_cfg, dataset_path):
    """
    :param save_root_path: save file destination path
    :param model_name: select preprocessing method
    :param data_root_path: data set root path
    :param dataset_name: data set name(ex. UBFC, COFACE)
    :return:
    """

    chunk_size = preprocess_cfg.chunk_size
    dataset_path = dataset_path
    manager = multiprocessing.Manager()
    for dataset in preprocess_cfg.datasets:
        dataset_name = dataset['name']
        if dataset['type'] in ['continuous', 'CONT']:
            preprocess_type = 'CONT'
        elif dataset['type'] in ['diff', 'DIFF']:
            preprocess_type = 'DIFF'
        else:
            preprocess_type = 'CUSTOM'

        img_size = dataset['image_size']
        large_box_coef = dataset['large_box_coef']

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
                task_list = [task for task in os.listdir(dataset_root_path + "/" + subject) if
                             task.__contains__('.mat')]
                for task in task_list:
                    tmp = subject + "/" + task
                    data_list.append(tmp)
            vid_name = ""
            ground_truth_name = ""

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
                                dataset_name, dataset_path, img_size=img_size, large_box_coef=large_box_coef)


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
    video_path = path + vid_name
    label_path = path + ground_truth_name

    raw_video, preprocessed_label = data_preprocess(preprocess_type, video_path, label_path, **kwargs)

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

    data = h5py.File(dir_path + path[-1] + ".hdf5", "w")
    data.create_dataset('raw_video', data=raw_video)
    data.create_dataset('preprocessed_label', data=preprocessed_label)
    data.close()


def chunk_preprocessing(preprocess_type, data_list, dataset_root_path, vid_name, ground_truth_name, dataset_name,
                        dataset_path, img_size, large_box_coef):
    process = []
    save_root_path = dataset_path

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    for index, data_path in enumerate(data_list):
        proc = multiprocessing.Process(target=preprocess_Dataset,
                                       args=(preprocess_type, dataset_root_path + "/" + data_path, vid_name,
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

    if video_path.__contains__("png"):
        path = video_path[:-4]
        data = sorted(os.listdir(path))[1:]
        frame_total = len(data)
        raw_label = get_label(label_path, frame_total)

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
                raw_video[frame_num] = face
            else:
                raw_video[frame_num] = cv2.resize(face, (img_size, img_size), interpolation=cv2.INTER_AREA)

    elif video_path.__contains__(".mat"):
        f = sio.loadmat(video_path)
        frames = f['video']
        raw_label = f['GT_ppg']
        frame_total = len(frames)

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

    else:
        cap = cv2.VideoCapture(video_path)
        frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        raw_label = get_label(label_path, frame_total)

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

    if preprocess_type == 'DIFF':
        raw_video = diff_normalize_label(raw_video)
        raw_label = diff_normalize_video(raw_label)
    elif preprocess_type == 'CUSTOM':
        pass

    return raw_video, raw_label


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