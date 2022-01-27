import numpy as np
import scipy.signal
from matplotlib import pyplot as plt
from scipy.signal import butter
from scipy.sparse import spdiags
import torch
import h5py

def detrend(signal, Lambda):
    """detrend(signal, Lambda) -> filtered_signal
    This function applies a detrending filter.
    This  is based on the following article "An advanced detrending method with application
    to HRV analysis". Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.
    *Parameters*
      ``signal`` (1d numpy array):
        The signal where you want to remove the trend.
      ``Lambda`` (int):
        The smoothing parameter.
    *Returns*
      ``filtered_signal`` (1d numpy array):
        The detrended signal.
    """
    signal_length = len(signal)

    # observation matrix
    H = np.identity(signal_length)

    # second-order difference matrix

    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot((H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), signal)
    return filtered_signal


def BPF(input_val, fs=30):
    low = 0.75 / (0.5 * fs)
    high = 2.5 / (0.5 * fs)
    [b_pulse, a_pulse] = butter(1, [low, high], btype='bandpass')
    return scipy.signal.filtfilt(b_pulse, a_pulse, np.double(input_val))


def plot_graph(start_point, length, target, inference):
    plt.rcParams["figure.figsize"] = (14, 5)
    plt.plot(range(len(target[start_point:start_point + length])), target[start_point:start_point + length],
             label='target')
    plt.plot(range(len(inference[start_point:start_point + length])), inference[start_point:start_point + length],
             label='inference')
    plt.legend(fontsize='x-large')
    plt.show()


def normalize(input_val):
    return (input_val - np.mean(input_val)) / np.std(input_val)

def getdatasets(key,archive):

  if key[-1] != '/': key += '/'

  out = []

  for name in archive[key]:

    path = key + name

    if isinstance(archive[path], h5py.Dataset):
      out += [path]
    else:
       out += getdatasets(path,archive)

  return out

def make_mixed_dataset(h5py_data, new_h5py_data):


    # open HDF5-files
    #data     = h5py.File('/media/hdd1/yj/dataset2/MetaPhysNet_V4V_train.hdf5','r')
    #new_data = h5py.File('/media/hdd1/yj/dataset2/MetaPhysNet_all_train.hdf5','a')
    data     = h5py.File(h5py_data,'r')
    new_data = h5py.File(new_h5py_data,'a') #all

    # read as much datasets as possible from the old HDF5-file
    datasets = getdatasets('/',data)

    # get the group-names from the lists of datasets
    groups = list(set([i[::-1].split('/',1)[1][::-1] for i in datasets]))
    groups = [i for i in groups if len(i)>0]

    # sort groups based on depth
    idx    = np.argsort(np.array([len(i.split('/')) for i in groups]))
    groups = [groups[i] for i in idx]

    # create all groups that contain dataset that will be copied
    for group in groups:
      new_data.create_group(group)

    # copy datasets
    for path in datasets:

      # - get group name
      group = path[::-1].split('/',1)[1][::-1]

      # - minimum group name
      if len(group) == 0: group = '/'

      # - copy data
      data.copy(path, new_data[group])