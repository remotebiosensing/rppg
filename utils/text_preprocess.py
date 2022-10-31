from sklearn.preprocessing import minmax_scale
import h5py
import csv
import json

import cv2
import h5py
import math
# from __future__ import absolute_import, division, print_function
# 3rd party
import numpy as np
from biosppy.signals import bvp
from scipy.signal import resample_poly
from sklearn.preprocessing import minmax_scale
import pandas as pd

# local


def Deepphys_preprocess_Label(path):

    if path.__contains__("label.txt"):
        cap = cv2.VideoCapture(path[:-9] + "video.mkv")
        frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

        length = int(frame_total / fps * 1000)

        f = open(path, 'r')
        f_read = f.read().split('\n')
        f.close()
        label = list(map(float, f_read[:-1]))
        new_label = label[:length:40]
        label = new_label
    else:
        f = open(path, 'r')
        f_read = f.read().split('\n')
        label = ' '.join(f_read[0].split()).split()
        label_hr = ' '.join(f_read[1].split()).split()
        f.close()

    '''
    :param path: label file path
    :return: delta pulse
    '''
    # TODO : need to check length with video frames
    # TODO : need to implement piecewise cubic Hermite interpolation
    # Load input
    label = list(map(float, label))
    delta_label = []
    for i in range(len(label) - 1):
        delta_label.append(label[i + 1] - label[i])
    delta_label -= np.mean(delta_label)
    delta_label /= np.std(delta_label)
    delta_label = np.array(delta_label).astype('float32')
    delta_pulse = delta_label.copy()  # 이거 왜 있지?
    split_hr_label = np.zeros(shape=delta_pulse.shape)

    return delta_pulse,split_hr_label


def PhysNet_preprocess_Label(path):
    '''
    :param path: label file path
    :return: wave form
    '''
    set = 64
    div = 64
    # Load input
    if path.__contains__("hdf5"):
        f = h5py.File(path, 'r')
        label = np.asarray(f['pulse'])

        # label = decimate(label,int(len(label)/frame_total))
        label_bvp = bvp.bvp(label, 256, show=False)
        label = label_bvp['filtered']

        label = smooth(label, 128)
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

    elif path.__contains__("json"):
        name = path.split("/")
        label = []
        label_time = []
        label_hr = []
        time = []
        with open(path[:-4] + name[-2] + ".json") as json_file:
            json_data = json.load(json_file)
            for data in json_data['/FullPackage']:
                label.append(data['Value']['waveform'])
                label_time.append(data['Timestamp'])
                label_hr.append(data['Value']['pulseRate'])
            for data in json_data['/Image']:
                time.append(data['Timestamp'])
            print(str(len(label)) + path)

        label_std = label_time[0]
        time_std = time[0]

        if label_std < time_std:
            time = [(i - label_std) / 1000 for i in time]
            label_time = [(i - label_std) / 1000 for i in label_time]
            chek = "레이블기준"

            j = 0
            i = 0
            new_label = []
            new_hr = []

            while i < len(time):
                if j + 1 >= len(label_time):
                    break
                if i == 0:
                    if time[i] <= label_time[j]:
                        new_label.append(0)
                        new_hr.append(0)
                        i += 1

                if label_time[j + 1] >= time[i] >= label_time[j]:
                    term = label_time[j + 1] - label_time[j]
                    head = time[i] - label_time[j]  # 앞에꺼
                    back = label_time[j + 1] - time[i]  # 뒤에꺼
                    new_label.append((label[i] * back + label[i - 1] * head) / term)
                    new_hr.append((label_hr[i] * back + label_hr[i - 1] * head) / term)
                    i += 1
                else:
                    j += 1

        else:
            label_time = [(i - time_std) / 1000 for i in label_time]
            time = [(i - time_std) / 1000 for i in time]
            chek = "시간기준"
            j = 0
            i = 0
            new_label = []
            new_hr = []

            while i < len(time):
                if j + 1 >= len(label_time):
                    break

                if time[i] <= label_time[j]:
                    new_label.append(0)
                    new_hr.append(0)
                    i += 1
                    continue

                if label_time[j + 1] >= time[i] and time[i] >= label_time[j]:
                    term = label_time[j + 1] - label_time[j]
                    head = time[i] - label_time[j]  # 앞에꺼
                    back = label_time[j + 1] - time[i]  # 뒤에꺼
                    new_label.append((label[i] * back + label[i - 1] * head) / term)
                    new_hr.append((label_hr[i] * back + label_hr[i - 1] * head) / term)
                    i += 1
                j += 1

        # test = nk.ppg_clean(label, 60)
        print("A")
        # label = resample(label,len(label)//2)
    elif path.__contains__("csv"):
        # print(path)
        interval = 20 * 1.2

        f = open(path, 'r')
        rdr = csv.reader(f)
        fr = list(rdr)
        label = np.asarray(fr[1:]).reshape((-1)).astype(np.float)
        # print("label length" + str(len(label)))
        f.close()

        f_time = open(path[:-8] + "time.txt", 'r')
        fr_time = f_time.read().split('\n')
        time = np.asarray(fr_time[:-1]).astype(np.float) * 1.2  ## 동영상 시간
        f_time.close()

        f_hr = open(path[:-8] + "gt_HR.csv", 'r')
        rdr = csv.reader(f_hr)
        fr = list(rdr)
        hr = np.asarray(fr[1:]).reshape((-1)).astype(np.float)
        f_hr.close()

        new_label = []
        new_hr = []
        j = 0
        i = 0
        while i <= len(label) - 1:

            if j >= len(time):
                break

            if time[j] == 0:
                new_label.append(label[0])
                new_hr.append(hr[0])
                if time[j + 1] > interval:
                    i += 1
                j += 1

            elif i * interval <= time[j] and time[j] <= (i + 1) * interval:
                # term =
                head = time[j] - i * interval
                back = (i + 1) * interval - time[j]
                new_label.append((label[i + 1] * back + label[i] * head) / interval)
                j += 1
                i += 1
            else:
                i += 1

                # head = time[j]%1000
                # back = 1000 - head
                # std = int(time[j]//1000)
                # # print("hr len" + str(len(hr)) + "std" +str(std))
                # if (std) >= len(hr):
                #     new_hr.append(hr[-1])
                # else:
                #     new_hr.append((hr[std] * back + hr[std+1] * head) / 1000)

        # print("Check")

    elif path.__contains__("label.txt"):
        cap = cv2.VideoCapture(path[:-9] + "video.mkv")
        frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

        length = int(frame_total / fps * 1000)

        f = open(path, 'r')
        f_read = f.read().split('\n')
        f.close()
        label = list(map(float, f_read[:-1]))
        new_label = label[:length:40]
        print("A")
    else:
        f = open(path, 'r')
        f_read = f.read().split('\n')
        label = ' '.join(f_read[0].split()).split()
        label_hr = ' '.join(f_read[1].split()).split()

        new_label = list(map(float, label))
        new_hr = list(map(float, label_hr))
        new_label = np.array(label).astype('float64')
        f.close()

    if (len(new_label) - 22) < 0:
        print("negative" + path)

    split_raw_label = np.zeros(((len(new_label) - 22) // 10, 32))
    split_hr_label = np.zeros(((len(new_label) - 22) // 10))

    index = 0
    # print( str(len(new_label)) + "       "+ str((len(new_label) -22) //10))
    for i in range((len(new_label) - 22) // 10):
        split_raw_label[i] = new_label[index:index + 32]
        # split_hr_label[i] = np.mean(new_hr[index:index + 32])
        index = index + 10

    return split_raw_label, split_hr_label


def GCN_preprocess_Label(path, sliding_window_stride):
    '''
    :param path: label file path
    :return: wave form
    '''

    div = 256
    stride = sliding_window_stride
    # Load input
    f = open(path, 'r')
    f_read = f.read().split('\n')
    label = ' '.join(f_read[0].split()).split()
    label = list(map(float, label))
    label = np.array(label).astype('float32')
    num_maps = int((len(label) - div) / stride + 1)
    split_raw_label = np.zeros((num_maps, div))
    index = 0
    for i in range(0, num_maps, stride):
        split_raw_label[i] = label[index:index + div]
        index = index + stride
    f.close()

    return split_raw_label


def Axis_preprocess_Label(path, sliding_window_stride, num_frames, clip_size=256):
    '''
    :param path: label file path
    :return: wave form
    '''

    # div = 256
    # stride = num_maps
    # Load input
    ext = path.split('.')[-1]

    f = open(path, 'r')

    f_read = f.read().split('\n')
    if ext == 'txt':
        label = ' '.join(f_read[0].split()).split()
        label = list(map(float, label))


    elif ext == 'csv':
        label = f_read[1:]
        label = [float(txt) for txt in label if txt != '']

    label = np.array(label).astype('float32')
    label = np.resize(label, num_frames)
    # print(path + str(len(label))+ "  " + str(num_maps)+"  "+str(clip_size) +"  " + str(sliding_window_stride) + "  " + str(num_frames))
    # print(num_maps)
    num_maps = int((num_frames - clip_size) / sliding_window_stride + 1)
    split_raw_label = np.zeros((num_maps, clip_size))
    index = 0
    for start_frame_index in range(0, num_frames, sliding_window_stride):
        end_frame_index = start_frame_index + clip_size
        if end_frame_index > num_frames:
            break
        split_raw_label[index, :] = minmax_scale(label[start_frame_index:end_frame_index], axis=0,
                                                 copy=True) * 2 - 1  # label[start_frame_index:end_frame_index]
        index += 1
    f.close()

    return split_raw_label

def RhythmNet_preprocess_Label(path, time_length=300):
    f = open(path, 'r')
    hr_list = f.read().split('\n')
    hr_list = [hr.strip() for hr in hr_list if hr != '']
    hr_list = list(map(float, hr_list))

    hr_mean = np.zeros((len(hr_list))//time_length)
    # 프레임 수가 같다고 가정 후 진행
    for i in range((len(hr_list))//time_length):
        hr_mean[i] = np.mean(hr_list[i*time_length:(i+1)*time_length])
    return hr_mean


def ETArPPGNet_preprocess_Label(path, time_length):
    time_length *= 30
    f = open(path, 'r')
    f_read = f.read().split('\n')
    label_hr = ' '.join(f_read[1].split()).split()
    new_hr = list(map(float, label_hr))
    new_hr = np.array(new_hr).astype('float64')
    new_hr = new_hr[:(len(new_hr)//time_length)*time_length].reshape(-1, time_length)
    f.close()
    return new_hr

