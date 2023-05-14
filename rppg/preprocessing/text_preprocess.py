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
from scipy.signal import resample_poly,resample
from scipy.interpolate import interp1d
from sklearn.preprocessing import minmax_scale
import pandas as pd

# local
def label_preprocess(preprocess_type, path,**kwargs):
    if preprocess_type == "DIFF":
        return Deepphys_preprocess_Label(path, **kwargs)
    elif preprocess_type == 'CONT':
        return PhysNet_preprocess_Label(path, **kwargs)
    else:
        raise ValueError('model_name is not valid')

def Deepphys_preprocess_Label(path, **kwargs):

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
def PhysNet_preprocess_Label(path, **kwargs):
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


    elif path.__contains__("csv"):
        f = open(path, 'r')
        rdr = csv.reader(f)
        fr = list(rdr)
        label = np.asarray(fr[1:]).reshape((-1)).astype(np.float)

        # print("label length" + str(len(label)))
        f.close()

        f_time = open(path[:-8] + "time.txt", 'r')
        fr_time = f_time.read().split('\n')
        time = np.asarray(fr_time[:-1]).astype(np.float)  ## 동영상 시간
        f_time.close()

        x = np.linspace(time[0],time[-1], len(label))
        new_x = np.linspace(time[0],time[-1], len(time))
        f = interp1d(x, label)
        label = f(new_x)

        f_hr = open(path[:-8] + "gt_HR.csv", 'r')
        rdr = csv.reader(f_hr)
        fr = list(rdr)
        label_hr = np.asarray(fr[1:]).reshape((-1)).astype(np.float)
        f_hr.close()

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
        label = list(map(float, label))
        label = np.array(label).astype('float32')
        label_hr = list(map(float, label_hr))
        label_hr = np.array(label_hr).astype('int')
        f.close()


    return label, label_hr
def GCN_preprocess_Label(path, **kwargs):
    '''
    :param path: label file path
    :return: wave form
    '''

    sliding_window_stride = kwargs['sliding_window_stride']

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
def Axis_preprocess_Label(path, **kwargs):
    '''
    :param path: label file path
    :return: wave form
    '''

    # div = 256
    # stride = num_maps
    # Load input

    num_frames = kwargs['num_frames']
    clip_size = kwargs['clip_size'] # 256
    sliding_window_stride = kwargs['sliding_window_stride']


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

def RhythmNet_preprocess_Label(path, **kwargs):
    time_length = kwargs['time_length']
    f = open(path, 'r')
    hr_list = f.read().split('\n')
    hr_list = [hr.strip() for hr in hr_list if hr != '']
    hr_list = list(map(float, hr_list))

    hr_mean = np.zeros((len(hr_list))//time_length)
    # 프레임 수가 같다고 가정 후 진행
    for i in range((len(hr_list))//time_length):
        hr_mean[i] = np.mean(hr_list[i*time_length:(i+1)*time_length])
    return hr_mean


def ETArPPGNet_preprocess_Label(path, **kwargs):
    time_length = kwargs['time_length']
    time_length *= 30
    f = open(path, 'r')
    f_read = f.read().split('\n')
    label_hr = ' '.join(f_read[1].split()).split()
    new_hr = list(map(float, label_hr))
    new_hr = np.array(new_hr).astype('float64')
    new_hr = new_hr[:(len(new_hr)//time_length)*time_length].reshape(-1, time_length)
    f.close()
    return new_hr


def Vitamon_preprocess_Label(path, time_length):
    f = open(path, 'r')
    f_read = f.read().split('\n')
    f_read = f_read[1:-1]
    new_hr = list(map(float, f_read))
    new_hr = np.array(new_hr).astype('float64')
    path = path[:-8] + 'video.avi'
    cap = cv2.VideoCapture(path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    new_hr = resample(new_hr, length)
    new_hr = new_hr[:(len(new_hr)//time_length)*time_length].reshape(-1, time_length)
    f.close()
    return new_hr