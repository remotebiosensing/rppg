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
def label_preprocess(preprocess_type, path, frame_total,**kwargs):
    if preprocess_type == "DIFF":
        return Deepphys_preprocess_Label(path,frame_total, **kwargs)
    elif preprocess_type == 'CONT':
        return PhysNet_preprocess_Label(path,frame_total, **kwargs)
    else:
        raise ValueError('model_name is not valid')

def Deepphys_preprocess_Label(path,frame_total, **kwargs):

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
    if len(label) != frame_total:
        label = np.interp(
            np.linspace(
                1, len(label), frame_total), np.linspace(
                1, len(label), len(label)), label)
    delta_label = []
    delta_label = np.diff(label, axis=0)
    delta_label /= np.std(delta_label)
    delta_label = np.array(delta_label).astype('float32')
    delta_label = np.append(delta_label, np.zeros(1), axis=0)
    delta_label[np.isnan(delta_label)] = 0
    split_hr_label = np.zeros(shape=delta_label.shape)

    return delta_label,split_hr_label
def PhysNet_preprocess_Label(path,frame_total, **kwargs):
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

    label = list(map(float, label))
    if len(label) != frame_total:
        label = np.interp(
            np.linspace(
                1, len(label), frame_total), np.linspace(
                1, len(label), len(label)), label)
    label = label - np.mean(label)
    label = label / np.std(label)


    return label, label_hr