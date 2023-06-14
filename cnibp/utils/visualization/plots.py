import json
import numpy as np
import matplotlib.pyplot as plt

with open('/home/paperc/PycharmProjects/Pytorch_rppgs/cnibp/configs/parameter.json') as f:
    json_data = json.load(f)
    param = json_data.get("parameters")
    channels = json_data.get("parameters").get("in_channels")
    hyper_param = json_data.get("hyper_parameters")
    wb = json_data.get("wandb")
    root_path = json_data.get("parameters").get("root_path")  # mimic uci containing folder
    data_path = json_data.get("parameters").get("dataset_path")  # raw mimic uci selection
    sampling_rate = json_data.get("parameters").get("sampling_rate")
    chunk_size = json_data.get("parameters").get("chunk_size")

dataset = "uci"
# channel = channels["third"]
samp_rate = sampling_rate["60"]
read_path = root_path + data_path[dataset][1]


def AHA_criteria():
    '''AHA'''
    # '''Normal'''
    # if (size_factor[1] < 120) and (size_factor[0] < 80):
    # '''Elevated'''
    # if (120 <= size_factor[1] < 130) and (size_factor[0] < 80):
    # '''Hypertension Stage 1'''
    # if (130 <= size_factor[1] < 140) or (80 <= size_factor[0] < 90):
    # '''Hypertension Stage 2'''
    # if (140 <= size_factor[1]) or (90 <= size_factor[0]):
    # '''Hypertensive Crisis'''
    # if (180 <= size_factor[1]) or (120 <= size_factor[0]):


# '''train dataset load'''
# with h5py.File(
#         "/home/paperc/PycharmProjects/VBPNet/dataset/BPNet_uci/corr0.7/case(f+f'+f'')_len(3)_360_train(True)_checker.hdf5",
#         "r") as train_f:
#     print("<train dataset>")
#     train_ple, train_abp, train_size = np.array(train_f['ple']), np.array(train_f['abp']), np.array(train_f['size'])
#     print(len(train_size))
# '''test dataset load'''
# with h5py.File(
#         "/home/paperc/PycharmProjects/VBPNet/dataset/BPNet_uci/corr0.7/case(f+f'+f'')_len(1)_360_train(False)_checker.hdf5",
#         "r") as test_f:
#     print("<test dataset>")
#     test_ple, test_abp, test_size = np.array(test_f['ple']), np.array(test_f['abp']), np.array(test_f['size'])
#     print(len(test_size))


# print(test_size[0])


def distribution_checker(train_data, test_data):
    pltname = 'Preprocessed ABP Signal Distribution\n'
    print(np.shape(train_data), np.shape(test_data))
    train_dbp_l = np.zeros((250,), dtype=int)
    test_dbp_l = np.zeros((250,), dtype=int)
    train_sbp_l = np.zeros((250,), dtype=int)
    test_sbp_l = np.zeros((250,), dtype=int)
    train_mbp_l = np.zeros((250,), dtype=int)
    test_mbp_l = np.zeros((250,), dtype=int)

    for tr in train_data:
        train_dbp_l[round(tr[0])] += 1
        train_sbp_l[round(tr[1])] += 1
        train_mbp_l[round(tr[2])] += 1
    for te in test_data:
        test_dbp_l[round(te[0])] += 1
        test_sbp_l[round(te[1])] += 1
        test_mbp_l[round(te[2])] += 1
    plt.title(pltname)
    plt.plot(train_dbp_l, color='#0066ff', linestyle='solid', linewidth=2, label='Train Diastolic')
    plt.plot(test_dbp_l, color='#0066ff', linestyle='dashed', linewidth=2, label='Test Diastolic')
    plt.plot(train_sbp_l, color='#ff5050', linestyle='solid', linewidth=2, label='Train Systolic')
    plt.plot(test_sbp_l, color='#ff5050', linestyle='dashed', linewidth=2, label='Test Systolic')
    plt.plot(train_mbp_l, color='#33cc33', linestyle='solid', linewidth=2, label='Train Mean')
    plt.plot(test_mbp_l, color='#33cc33', linestyle='dashed', linewidth=2, label='Test Mean')
    plt.xlabel('Arterial Blood Pressure ( mmHg )')
    plt.ylabel('number of data')
    plt.legend()
    plt.show()

    print()


# distribution_checker(train_size, test_size)


def ppg_plot(ppg_data):
    # pltname = 'Preprocessed PPG Signal\n'
    # plt.title(pltname)
    # plt.plot(ppg_data[110:169],'r')
    plt.plot(ppg_data, 'g', label='VPG : PPG(t+1) - PPG(t)')
    # plt.plot(ppg_data[1], label='First Derivative of PPG')
    # plt.plot(ppg_data[2], label='Second Derivative of PPG')
    # plt.ylabel('Arterial Blood Pressure (mmHg)')
    plt.xlabel('Time')
    # plt.axhline(145,color='gray',linestyle='dashed', label='Systolic Blood Pressure')
    # plt.axhline(56,color='lightgray',linestyle='dashdot', label='Diastolic Blood Pressure')
    plt.legend()
    plt.show()


# ppg_d = test_ple[0][0][51:85]
# ppg_c = test_ple[0][0][50:84]
# ppg_diff = ppg_d-ppg_c
# ppg_plot(ppg_diff)

def dicrotic_fft(input_sig):
    '''
    https://lifelong-education-dr-kim.tistory.com/4
    '''
    Fs = 60
    T = 1 / Fs
    end_time = 1
    time = np.linspace(0, end_time, Fs)
    abnormal_cnt = 0
    flag = False

    plt.subplot(4, 1, 1)
    plt.plot(input_sig)
    plt.title("original signal")
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.70)

    s_fft = np.fft.fft(input_sig)  # 추후 IFFT를 위해 abs를 취하지 않은 값을 저장한다.
    amplitude = abs(s_fft) * (2 / len(s_fft))  # 2/len(s)을 곱해줘서 원래의 amp를 구한다.
    frequency = np.fft.fftfreq(len(s_fft), T)

    plt.subplot(4, 1, 2)
    plt.xlim(0, 30)
    plt.stem(frequency, amplitude)
    plt.grid(True)
    plt.title("fft result")
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.70)

    # # Dicrotic Notch amplify
    fft_freq = frequency.copy()
    # dicrotic_peak_index0 = amplitude[:int(len(amplitude) / 2)].argsort()[-1]
    dicrotic_peak_index1 = amplitude[:int(len(amplitude) / 2)].argsort()[-2]
    dicrotic_peak_index2 = amplitude[:int(len(amplitude) / 2)].argsort()[-3]
    peak_freq = fft_freq[dicrotic_peak_index1]
    #
    fft_2x = s_fft.copy()
    # fft_2x[dicrotic_peak_index0] *= 2.0
    # fft_2x[dicrotic_peak_index1] *= 0.8
    fft_2x[dicrotic_peak_index2] *= 1.5
    amplitude_2x = abs(fft_2x) * (2 / len(fft_2x))
    plt.subplot(4, 1, 3)
    plt.xlim(0, 30)
    plt.stem(frequency, amplitude_2x)
    plt.grid(True)
    plt.title("Notch frequency amplified")
    filtered_data = np.fft.ifft(fft_2x)
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.70)

    cycle_len = round(Fs / peak_freq)
    # cycle_min = int(np.where(np.min(input[:cycle_len])))
    plt.subplot(4, 1, 4)
    plt.title("Dicrotic Notch amplified signal")
    plt.plot(filtered_data)

    # plt.plot(filtered_data[:cycle_len], color='indigo')
    # plt.plot(input_sig)
    # plt.title("amplified signal")
    # plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.70)

    plt.show()
    print("break point")
    return input_sig


# def abp_plot(abp_data)

# ppg_t = test_ple[6877]
# fft_result = dicrotic_fft(ppg_t[0])



# train_test_shuffler(train_ple, test_ple, train_abp, test_abp, train_size, test_size)


