import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

'''should be done before sig_slicing'''


def derivative(input_sig):
    # print('input_sig :', np.shape(input_sig))
    # print('input_sig :', input_sig)
    # deriv = np.array(len(input_sig))
    temp = input_sig[1:]
    # print('input_sig[1:]', np.shape(temp))
    # print('input_sig[1:]', temp)
    temp = np.append(temp, temp[-1])
    # print('temp', np.shape(temp))
    # print('temp', temp)
    # print('np.append(temp,0)', temp)
    # deriv = np.subtract(temp, input_sig)
    deriv = [tempi - inputi for tempi, inputi in zip(temp, input_sig)]
    # print('deriv :', np.shape(deriv))
    # print('deriv :', deriv)
    return deriv


'''
np.shape(input_sig) : ndarray(702, 7500)     ex) ple, abp
'''



def diff_np(input_sig):
    diff = []
    # print('before diff_np :', np.shape(input_sig))
    for s in input_sig:
        temp = np.append(s[1:], s[-1]) - s
        temp[-1] = np.mean(temp[-3:-2])
        diff.append(temp)
    diff = np.array(diff)
    # print('before diff_np :', np.shape(diff))
    return diff


def diff_channels_aggregator(zero, first=None, second=None):
    zero = np.expand_dims(zero, axis=1)

    if first is None:
        # print('zero called')
        print(np.shape(zero))
        print(zero)
        return zero

    elif (first is not None) and (second is None):
        # print('first called')
        first = np.expand_dims(first, axis=1)
        temp1 = np.concatenate((zero, first),axis=1)

        print(np.shape(temp1))
        print(temp1)
        return temp1
    elif (first is not None) and (second is not None):
        # print('second called')
        first = np.expand_dims(first, axis=1)
        second = np.expand_dims(second, axis=1)
        temp2 = np.concatenate((zero, first, second), axis=1)

        print(np.shape(temp2))
        print(temp2)
        return temp2

