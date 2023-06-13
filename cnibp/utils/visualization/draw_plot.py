import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm


def dbp_sbp_scatter_plot(ch):
    with open('/home/paperc/PycharmProjects/Pytorch_rppgs/cnibp/configs/parameter.json') as f:
        json_data = json.load(f)
        channels = json_data.get("parameters").get("in_channels")
        root_path = json_data.get("parameters").get("root_path")  # mimic uci containing folder
        data_path = json_data.get("parameters").get("dataset_path")  # raw mimic uci selection

    dataset = 'uci'
    # read_path == dataset/BPNet_uci
    read_path = root_path + data_path[dataset][1]
    channel = channels[ch][-1]
    '''test dataset load'''
    print(read_path)
    with h5py.File(
            read_path + "shuffled/case(" + str(channel) + ")_360_train(False)_checker_shuffled.hdf5",
            "r") as test_f:
        print("<test dataset>")
        # np.shape(test_ple) (43825, 3 ,360), np.shape(test_abp) (43825, 360), np.shape(test_size) (43825, 3)
        test_ple, test_abp, test_size = np.array(test_f['ple']), np.array(test_f['abp']), np.array(test_f['size'])
        print(len(test_size))

    # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_root_path = '/home/paperc/PycharmProjects/Pytorch_rppgs/cnibp/weights/BPNet_V1.0/'
    model = torch.load(model_root_path + "model_uci_(" + str(channel) + ")_bpnet.pt")
    model = model.to('cpu')
    model.eval()
    x = np.array(range(40, 200))
    np.random.seed(0)

    plt.figure(figsize=(6, 6))
    for t, s in zip(torch.Tensor(test_ple[:1000]), tqdm(torch.Tensor(test_size[:1000]))):
        pred_d = torch.squeeze(model(t)[1]).__float__()
        pred_s = torch.squeeze(model(t)[2]).__float__()
        plt.scatter(s[0], pred_d, c='blue', marker='x', alpha=0.1)
        plt.scatter(s[1], pred_s, c='red', alpha=0.1)

    plt.title(str(channel) + ' model prediction\n')

    plt.plot(x, x, color='k', label='y=x')
    plt.grid(color='gray', alpha=.5, linestyle='--')
    plt.xlabel('Target BP')
    plt.ylabel('Predicted BP')
    plt.legend()
    plt.show()

#
# dbp_sbp_scatter_plot('zero')
# dbp_sbp_scatter_plot('third')
dbp_sbp_scatter_plot('sixth')


def AHA_class_scatter_plot():
    with open('/home/paperc/PycharmProjects/Pytorch_rppgs/cnibp/configs/parameter.json') as f:
        json_data = json.load(f)
        channels = json_data.get("parameters").get("in_channels")
        root_path = json_data.get("parameters").get("root_path")  # mimic uci containing folder
        data_path = json_data.get("parameters").get("dataset_path")  # raw mimic uci selection

        dataset = 'uci'
        # read_path == dataset/BPNet_uci
        read_path = root_path + data_path[dataset][1]
        channel = channels['sixth'][-1]
        '''test dataset load'''
        with h5py.File(
                read_path + "AHAclass/case(" + str(channel) + ")_360_train(False)_1(normal).hdf5",
                "r") as test_f1:
            print("<test dataset>")
            # np.shape(test_ple) (43825, 3 ,360), np.shape(test_abp) (43825, 360), np.shape(test_size) (43825, 3)
            test_ple1, test_abp1, test_size1 = np.array(test_f1['ple']), np.array(test_f1['abp']), np.array(
                test_f1['size'])
            print(len(test_size1))
        with h5py.File(
                read_path + "AHAclass/case(" + str(channel) + ")_360_train(False)_2(elevated).hdf5",
                "r") as test_f2:
            print("<test dataset>")
            # np.shape(test_ple) (43825, 3 ,360), np.shape(test_abp) (43825, 360), np.shape(test_size) (43825, 3)
            test_ple2, test_abp2, test_size2 = np.array(test_f2['ple']), np.array(test_f2['abp']), np.array(
                test_f2['size'])
            print(len(test_size2))
        with h5py.File(
                read_path + "AHAclass/case(" + str(channel) + ")_360_train(False)_3(stage1).hdf5",
                "r") as test_f3:
            print("<test dataset>")
            # np.shape(test_ple) (43825, 3 ,360), np.shape(test_abp) (43825, 360), np.shape(test_size) (43825, 3)
            test_ple3, test_abp3, test_size3 = np.array(test_f3['ple']), np.array(test_f3['abp']), np.array(
                test_f3['size'])
            print(len(test_size3))
        with h5py.File(
                read_path + "AHAclass/case(" + str(channel) + ")_360_train(False)_4(stage2).hdf5",
                "r") as test_f4:
            print("<test dataset>")
            # np.shape(test_ple) (43825, 3 ,360), np.shape(test_abp) (43825, 360), np.shape(test_size) (43825, 3)
            test_ple4, test_abp4, test_size4 = np.array(test_f4['ple']), np.array(test_f4['abp']), np.array(
                test_f4['size'])
            print(len(test_size4))
        with h5py.File(
                read_path + "AHAclass/case(" + str(channel) + ")_360_train(False)_5(crisis).hdf5",
                "r") as test_f5:
            print("<test dataset>")
            # np.shape(test_ple) (43825, 3 ,360), np.shape(test_abp) (43825, 360), np.shape(test_size) (43825, 3)
            test_ple5, test_abp5, test_size5 = np.array(test_f5['ple']), np.array(test_f5['abp']), np.array(
                test_f5['size'])
            print(len(test_size5))

        # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_root_path = '/home/paperc/PycharmProjects/Pytorch_rppgs/cnibp/weights/BPNet_V1.0/'
        model = torch.load(model_root_path + "model_uci_(" + str(channel) + ")_checker_shuffled.pt")
        model = model.to('cpu')
        model.eval()
        x = np.array(range(40, 200))
        np.random.seed(0)

        plt.figure(figsize=(6, 6))
        for t, s in zip(torch.Tensor(test_ple1[:1900]), tqdm(torch.Tensor(test_size1[:1900]))):
            pred_d = torch.squeeze(model(t)[1]).__float__()
            pred_s = torch.squeeze(model(t)[2]).__float__()
            plt.scatter(s[0], pred_d, c='royalblue', marker='x', alpha=0.1)
            plt.scatter(s[1], pred_s, c='royalblue', alpha=0.1)
        for t, s in zip(torch.Tensor(test_ple2[:920]), tqdm(torch.Tensor(test_size2[:920]))):
            pred_d = torch.squeeze(model(t)[1]).__float__()
            pred_s = torch.squeeze(model(t)[2]).__float__()
            plt.scatter(s[0], pred_d, c='limegreen', marker='x', alpha=0.1)
            plt.scatter(s[1], pred_s, c='limegreen', alpha=0.1)
        for t, s in zip(torch.Tensor(test_ple3[:1400]), tqdm(torch.Tensor(test_size3[:1400]))):
            pred_d = torch.squeeze(model(t)[1]).__float__()
            pred_s = torch.squeeze(model(t)[2]).__float__()
            plt.scatter(s[0], pred_d, c='gold', marker='x', alpha=0.1)
            plt.scatter(s[1], pred_s, c='gold', alpha=0.1)
        for t, s in zip(torch.Tensor(test_ple4[:1700]), tqdm(torch.Tensor(test_size4[:1700]))):
            pred_d = torch.squeeze(model(t)[1]).__float__()
            pred_s = torch.squeeze(model(t)[2]).__float__()
            plt.scatter(s[0], pred_d, c='orangered', marker='x', alpha=0.1)
            plt.scatter(s[1], pred_s, c='orangered', alpha=0.1)
        for t, s in zip(torch.Tensor(test_ple5[:140]), tqdm(torch.Tensor(test_size5[:140]))):
            pred_d = torch.squeeze(model(t)[1]).__float__()
            pred_s = torch.squeeze(model(t)[2]).__float__()
            plt.scatter(s[0], pred_d, c='red', marker='x', alpha=0.1)
            plt.scatter(s[1], pred_s, c='red', alpha=0.1)

        plt.title(str(channel) + ' model AHA Class Prediction\n')

        plt.plot(x, x, color='k', label='y=x')
        plt.grid(color='gray', alpha=.5, linestyle='--')
        plt.xlabel('Target BP')
        plt.ylabel('Predicted BP')
        plt.legend()
        plt.show()


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
# 19205, 9204, 14390, 17961, 1407
# AHA_class_scatter_plot()
