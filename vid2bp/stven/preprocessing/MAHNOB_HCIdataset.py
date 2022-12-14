import pybdf
import numpy as np
import matplotlib.pyplot as plt
import vid2bp.preprocessing.utils.signal_utils as su
import heartpy as hp
from scipy.signal import resample
import os
from tqdm import tqdm

# 동영상은 30초 길이로 잘라야 함
# 동영상 fps는
# 10기준 영상은 00:02:16 ( 136초 )
# 10기준 ecg1은 166초 (sampling rate : 256)
'''
Remote Heart Rate Measurement from Highly Compressed Facial Videos: an
End-to-end Deep Learning Solution with Video Enhancement

used OBF dataset & MAHNOB-HCI dataset
                    ecg : EXG2 signal( channel 34 )
    
'''
samp_rate = 256
path = '/home/paperc/PycharmProjects/dataset/rppg/MAHNOB_HCI/'

file_name = '10/Part_1_S_Trial5_emotion.bdf'

file_p = path + file_name

'''
 read record for one folder
 return : list of n * [ecg slice for 30 second, good signal detection, heart rate]
'''


def read_record(file_path: str, signal: str = 'EXG2'):
    print(file_path)
    ecg_slice = []
    record = pybdf.bdfRecording(file_path)
    channel_label = record.chanLabels
    samp_rate = record.sampRate[channel_label.index(signal)]
    print('sampling rate of %s :' % signal, samp_rate)

    ecgGT = np.squeeze(record.getData(channels=[signal])['data'])
    # if len(ecgGT) % (samp_rate * 30) != 0:
    #     ecgGT = ecgGT[:-(len(ecgGT) % (samp_rate * 30))]
    # print(ecgGT.shape)

    for i in range(0, len(ecgGT), samp_rate * 30):
        # [30 second slice of ecg, good signal detection, heart rate]
        ecg_slice.append([ecgGT[i:i + samp_rate * 30], False, 0])
    for e in ecg_slice:
        filtered = hp.filter_signal(e[0], cutoff=0.05, sample_rate=samp_rate, filtertype='notch')
        resampled_data = resample(filtered, len(filtered) * 2)
        try:
            wd, m = hp.process(resampled_data, sample_rate=samp_rate * 2)
            # wd, m = hp.process_segmentwise(resampled_data, sample_rate=samp_rate * 2)
            # plt.figure(figsize=(12, 4))
            # hp.plotter(wd, m)

            e[1] = True
            e[2] = m['bpm']
            # print(len(m['bpm']))

        except hp.exceptions.BadSignalWarning:
            continue
        # print(len(ecg_slice))
    return ecg_slice


'''
read record for the whole dataset
return : list of m * (read_record())
'''


def data_aggregator(root_path):
    total_ecg = []
    cnt = 0
    for (path, dirs, files) in os.walk(root_path):
        cnt += 1
        if cnt < 5:
            for file in tqdm(files):
                if file.split('.')[-1] == 'bdf':
                    total_ecg.append(read_record(path + '/' + file))

    return total_ecg


total = data_aggregator(path)
# print(np.shape(total))
print(np.shape(total[0]))
print(np.shape(total[0][0]))
print('ecg GT :', np.shape(total[0][0][0]))
print('flag :', total[0][0][1])
print('bpm :', total[0][0][2])

# record = pybdf.bdfRecording(path + file_name)
# rec33 = record.getData(channels=['EXG1'])  # ECG1(Ch.33) (upper right corner of chest, under clavicle bone)
# rec34 = record.getData(channels=['EXG2'])  # ECG2(Ch.34) (upper left corner of chest, under clavicle bone)
# rec35 = record.getData(channels=['EXG3'])  # ECG3(Ch.35) (left side of abdomen)
# rec46 = record.getData(channels=['Resp'])  # Respiration(Ch.46) (Respiration belt)
# data31 = np.array(rec31['data'][0])[rec31['eventTable']['idx'][1]:rec31['eventTable']['idx'][4]]
# data33 = np.array(rec33['data'][0])[:]
# startT = record.startTime(['EXG2'])
# print(startT)
# samprate = rec34.sampRate()
# print(samprate)
# ecgGT = np.array(rec34['data'][0][41000:42250])
# peaktest, _= su.get_systolic(ecgGT)
# print(_)
# print(peaktest)
# print(len(peaktest))
# data35 = np.array(rec35['data'][0][:])
# data46 = np.array(rec46['data'][0][:])
# print(len(ecgGT))
# print(len(ecgGT) / 256)

# to get rr interval from ecg


#
# plt.plot(data31)
# plt.show()
# plt.plot(data33, 'r')
# plt.plot(ecgGT, 'g')
# plt.plot(data35, 'b')
# plt.plot(data46, 'y')
# plt.show()
# print(np.mean(data33))
# print(np.mean(data34))
# print(np.mean(data35))
# eventkey = record.getData['eventTable'].keys()
# 10의 끝은 42250
# 30초 * 256(sr)  =7680
# ecgtest = np.array(rec34['data'][0][0:2048])

# plt.figure(figsize=(12, 4))
# plt.plot(ecgtest, 'r')
#
# filtered = hp.filter_signal(ecgtest, cutoff=0.05, sample_rate=samp_rate, filtertype='notch')
# peaks, _ = su.get_systolic(filtered)
#
# wd, m = hp.process(hp.scale_data(filtered), samp_rate)
# plt.figure(figsize=(12, 4))
# hp.plotter(wd, m)
#
# for measure in m.keys():
#     print('%s: %f' % (measure, m[measure]))

# resampled_data = resample(filtered, len(filtered) * 2)
# wd, m = hp.process(hp.scale_data(resampled_data), samp_rate * 2)
# print(wd)
# print(m)
# plt.figure(figsize=(12, 4))
# hp.plotter(wd, m)
#
# for measure in m.keys():
#     print('%s: %f' % (measure, m[measure]))
