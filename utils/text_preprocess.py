import numpy as np
import h5py
import POS.pos as pos
import scipy.signal
import xml.etree.ElementTree as ET
import pandas as pd

def Deepphys_preprocess_Label(path):
    '''
    :param path: label file path
    :return: delta pulse
    '''
    # TODO : need to check length with video frames
    # TODO : need to implement piecewise cubic Hermite interpolation
    # Load input
    f = open(path, 'r')
    f_read = f.read().split('\n')
    label = ' '.join(f_read[0].split()).split()
    label = list(map(float, label))

    delta_label = []
    for i in range(len(label) - 1):
        delta_label.append(label[i + 1] - label[i])
    delta_label -= np.mean(delta_label)
    delta_label /= np.std(delta_label)
    delta_label = np.array(delta_label).astype('float32')
    delta_pulse = delta_label.copy()  # 이거 왜 있지?
    f.close()

    return delta_pulse


def PhysNet_preprocess_Label(path):
    '''
    :param path: label file path
    :return: wave form
    '''
    # Load input
    f = open(path, 'r')
    f_read = f.read().split('\n')
    label = ' '.join(f_read[0].split()).split()
    label = list(map(float, label))
    label = np.array(label).astype('float32')
    split_raw_label = np.zeros(((len(label) // 32), 32))
    index = 0
    for i in range(len(label) // 32):
        split_raw_label[i] = label[index:index + 32]
        index = index + 32
    f.close()

    return split_raw_label

def cohface_Label(path, frame_total):
    f = h5py.File(path, "r")
    label = list(f['pulse'])
    f.close()
    label = np.interp(np.arange(0, len(frame_total)+1),
                      np.linspace(0, len(frame_total)+1, num=len(label)),
                      label)
    delta_label = []
    for i in range(len(label) - 1):
        delta_label.append(label[i + 1] - label[i])
    delta_label -= np.mean(delta_label)
    delta_label /= np.std(delta_label)
    delta_label = np.array(delta_label).astype('float32')
    delta_pulse = delta_label.copy()  # 이거 왜 있지?

    return delta_pulse

def PhysNet_cohface_Label(path, frame_total):
    f = h5py.File(path, "r")
    label = list(f['pulse'])
    f.close()
    label = np.interp(np.arange(0, frame_total+1),
                      np.linspace(0, frame_total+1, num=len(label)),
                      label)

    split_raw_label = np.zeros(((len(label) // 32), 32))
    index = 0
    for i in range(len(label) // 32):
        split_raw_label[i] = label[index:index + 32]
        index = index + 32

    return split_raw_label

def LGGI_Label(path, frame_total):
    doc = ET.parse(path)
    root = doc.getroot()
    label = []

    for value in root:
        label.append(int(value.findtext('value2')))

    label = np.array(label).astype('float32')
    label = scipy.signal.resample(label, frame_total)

    split_raw_label = np.zeros(((len(label) // 32), 32))
    index = 0
    for i in range(len(label) // 32):
        split_raw_label[i] = label[index:index + 32]
        index = index + 32
    #print(len(split_raw_label))
    return split_raw_label

def V4V_Label(video, framerate):
    label = pos.PPOS(video, framerate)

    split_raw_label = np.zeros(((len(label) // 32), 32))
    index = 0
    for i in range(len(label) // 32):
        split_raw_label[i] = label[index:index + 32]
        index = index + 32

    return split_raw_label

def VIPL_Label(path, frame_total):
    f = pd.read_csv(path)
    label = f['Wave']
    label = np.array(label).astype('float32')
    label = scipy.signal.resample(label, frame_total)

    split_raw_label = np.zeros(((len(label) // 32), 32))
    index = 0
    for i in range(len(label) // 32):
        split_raw_label[i] = label[index:index + 32]
        index = index + 32
    #print(split_raw_label.shape)
    return split_raw_label