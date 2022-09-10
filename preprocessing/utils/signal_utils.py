import numpy as np
from scipy import signal
from sklearn import preprocessing
import matplotlib.pyplot as plt
import h5py
import json
from preprocessing import customdataset
from torch.utils.data import DataLoader
from scipy.misc import electrocardiogram

# from preprocessing import MIMICdataset


with open("/home/paperc/PycharmProjects/VBPNet/config/parameter.json") as f:
    json_data = json.load(f)
    param = json_data.get("parameters")
    channels = json_data.get("parameters").get("in_channels")
    sr = json_data.get("parameters").get("sampling_rate")
    hyper_param = json_data.get("hyper_parameters")
    wb = json_data.get("wandb")


# ''' train data load '''
# path = '/home/paperc/PycharmProjects/VBPNet/dataset/BPNet_mimic/raw.hdf5'
#
# # TODO use hdf5 file for training Done
# with h5py.File(path, "r") as f:
#     raw = np.array(f['raw'])
#
# print('rawdata :', np.shape(raw))


def sig_spliter(in_sig):
    if np.ndim(in_sig) == 2:
        # for mimicdataset (2) (ABP, PPG)
        if np.shape(in_sig)[-1] == 2:
            abp_split, ple_split = np.split(in_sig, 2, axis=1)
        # for ucidataset (3) (PPG, ABP, ECG)
        else:
            ple_split, abp_split, _ = np.split(in_sig, 3, axis=1)
        return abp_split, ple_split

    elif np.ndim(in_sig) == 3:  # ndim==3
        if np.shape(in_sig)[-1] == 2:
            abp_split, ple_split = np.split(in_sig, 2, axis=2)
            return abp_split, ple_split
        else:
            print("not supported shape for sig_spliter() due to different length of data")

    else:
        print("not supported dimension for sig_spliter")


'''
can be done either before or after sig_spliter()
'''


def down_sampling(in_signal, num=60):
    if num == 60:
        rst_sig = signal.resample(in_signal, int(param["chunk_size"] / 125) * sr["60"])
        return rst_sig
    elif num == 30:
        rst_sig = signal.resample(in_signal, int(param["chunk_size"] / 125) * sr["30"])
        return rst_sig
    else:
        print("not supported sampling rate..")
        return None


def scaler(input_sig, bottom, top):
    print(np.min(input_sig), '~', np.max(input_sig))
    input_sig = np.reshape(input_sig, (-1, 1))
    scaled_output = preprocessing.MinMaxScaler(feature_range=(bottom, top)).fit_transform(input_sig)
    print(np.min(scaled_output), '~', np.max(scaled_output))

    return scaled_output


# test_scaled = scaler(test)


def peak_detection(in_signal):
    # TODO SBP, DBP 구해야 함  SBP : Done
    x = np.squeeze(in_signal)
    mean = np.mean(x)
    print(mean)
    peaks, prop = signal.find_peaks(x, height=mean, distance=30)
    plt.plot(x)
    plt.plot(peaks, x[peaks], "x")
    # plt.plot(np.zeros_like(x), "--", color="gray")
    plt.show()
    # sbp = np.zeros(shape=(len(x),))
    # for i in range(len(prop["peak_heights"])):
    #     sbp[peaks[i]] = prop["peak_heights"][i]
    # return sbp

    return peaks, prop["peak_heights"], len(peaks)


# TODO IF STD(PEAK) IS MUCH LARGER THAN PRIOR SLICE, THEN DROP

def signal_slicing(rawdata):
    abp_list = []
    ple_list = []
    if np.shape(rawdata[0]) != (750, 2):
        rawdata = np.reshape(rawdata, (-1, 750, 2))
    cnt = 0
    from tqdm import tqdm
    for r in tqdm(rawdata):
        abp, ple = sig_spliter(r)
        # abp = down_sampling(np.squeeze(abp))
        # ple = down_sampling(np.squeeze(ple))
        abp = np.squeeze(abp)
        ple = np.squeeze(ple)
        p_abp, pro_abp = signal.find_peaks(abp, height=np.max(abp) - np.std(abp), distance=30)
        # abp_len = len(p_abp)
        p_ple, pro_ple = signal.find_peaks(ple, height=np.mean(ple), distance=30)
        # first_test = np.append(np.diff(ple), ple[-1])
        # p_first, pro_first = signal.find_peaks(first_test, height=np.mean(first_test), distance=30)
        # second_test = np.append(np.diff(first_test), first_test[-1])
        # p_second, pro_second = signal.find_peaks(second_test, height=np.mean(second_test), distance=30)
        if not ((np.mean(ple) == (0.0 or np.nan)) or (np.mean(abp) == 80.0) or
                (len(p_abp) < 7) or (len(p_ple) == 0) or
                (len(p_abp) - len(p_ple) > 2) or (np.std(pro_abp["peak_heights"]) > 5) or
                (np.mean(pro_abp["peak_heights"]) > 160)):
            cnt += 1
            # print(len(p_abp), len(p_ple), len(p_first), len(p_second), np.mean(abp), np.mean(ple))
            # print(np.max(abp), np.var(abp), np.std(abp), np.std(pro_abp["peak_heights"]))
            abp_list.append(abp)
            ple_list.append(ple)
    print(cnt, '/', len(rawdata))
    return abp_list, ple_list


# plt.plot(ple)
# plt.plot(p_ple, ple[p_ple], "o")
# #plt.plot(scaler(first_test.reshape(-1,1)))
# plt.plot(first_test)
# plt.plot(p_first, first_test[p_first], "x")
# plt.plot(second_test)
# plt.plot(p_second, second_test[p_second], "x")
# plt.plot(abp)
# plt.plot(p_abp, abp[p_abp], "x")
# #plt.plot(scaler(second_test.reshape(-1,1)))
# plt.show()


def mean_blood_pressure(dbp, sbp):
    return dbp + sbp / 3
