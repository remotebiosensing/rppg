import json
import numpy as np
import h5py
import matplotlib.pyplot as plt
import vid2bp.hdf_handler as hdf_handler

with open('/home/paperc/PycharmProjects/Pytorch_rppgs/vid2bp/config/parameter.json') as f:
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


def train_test_shuffler(tr_ple, te_ple, tr_abp, te_abp, tr_size, te_size):
    import random
    print('shuffler called')
    with open('/home/paperc/PycharmProjects/VBPNet/config/parameter.json') as f:
        json_data = json.load(f)
        root_path = json_data.get("parameters").get("root_path")  # mimic uci containing folder
        data_path = json_data.get("parameters").get("dataset_path")  # raw mimic uci selection
        orders = json_data.get("parameters").get("in_channels")

    order = orders["sixth"]
    write_path = root_path + data_path[dataset][1]

    ple_total = np.concatenate((tr_ple, te_ple), axis=0)
    abp_total = np.concatenate((tr_abp, te_abp), axis=0)
    size_total = np.concatenate((tr_size, te_size), axis=0)
    total_data = [[p, a, s] for p, a, s in zip(ple_total, abp_total, size_total)]
    random.shuffle(total_data)
    total_len = len(total_data)
    train_len = int(total_len * 0.8)
    shuffled_ple = np.squeeze([n[0] for n in total_data])
    shuffled_abp = np.squeeze([n[1] for n in total_data])
    shuffled_size = np.squeeze([n[2] for n in total_data])
    shuffled_tr_ple, shuffled_te_ple = np.split(shuffled_ple, [train_len])
    shuffled_tr_abp, shuffled_te_abp = np.split(shuffled_abp, [train_len])
    shuffled_tr_size, shuffled_te_size = np.split(shuffled_size, [train_len])
    print("train size :", len(shuffled_tr_ple), "test size :", len(shuffled_te_size))

    shuffled_train_dset = h5py.File(write_path + "/shuffled/case(" + str(order[-1]) + ")_"
                                    + str(int(param["chunk_size"] / sampling_rate["base"]) * samp_rate)
                                    + "_shuffled_train.hdf5", "w")
    shuffled_train_dset['ple'] = shuffled_tr_ple
    shuffled_train_dset['abp'] = shuffled_tr_abp
    shuffled_train_dset['size'] = shuffled_tr_size
    shuffled_train_dset.close()

    shuffled_test_dset = h5py.File(write_path + "/shuffled/case(" + str(order[-1]) + ")_"
                                   + str(int(param["chunk_size"] / sampling_rate["base"]) * samp_rate)
                                   + "_shuffled_test.hdf5", "w")
    shuffled_test_dset['ple'] = shuffled_te_ple
    shuffled_test_dset['abp'] = shuffled_te_abp
    shuffled_test_dset['size'] = shuffled_te_size
    shuffled_test_dset.close()

    distribution_checker(shuffled_tr_size, shuffled_te_size)


# train_test_shuffler(train_ple, test_ple, train_abp, test_abp, train_size, test_size)


def data_shuffler(model_name, save_path, input_data, cv):
    import random
    print('data shuffler called')
    ple = input_data[0]
    abp = input_data[1]
    if model_name == "BPNet":
        size = input_data[2]
        total_data = [[p, a, s] for p, a, s in zip(ple, abp, size)]
        random.shuffle(total_data)
        total_len = len(total_data)
        if cv == 1:
            train_val_len = int(total_len * 0.85)
        else:
            train_val_len = int(total_len * 0.8)
        shuffled_ple = np.squeeze([n[0] for n in total_data])
        shuffled_abp = np.squeeze([n[1] for n in total_data])
        shuffled_size = np.squeeze([n[2] for n in total_data])
        print('***********\nnp.shape(shuffled_ple) :', np.shape(shuffled_ple))
        shuffled_tr_ple, shuffled_te_ple = np.split(shuffled_ple, [train_val_len])
        shuffled_tr_abp, shuffled_te_abp = np.split(shuffled_abp, [train_val_len])
        shuffled_tr_size, shuffled_te_size = np.split(shuffled_size, [train_val_len])
        print('np.shape(shuffled_tr_ple) :', np.shape(shuffled_tr_ple))
        print('np.shape(shuffled_te_ple) :', np.shape(shuffled_te_ple))
        print('data shuffle for BPNet done')
        test_dset = h5py.File(save_path + "_test.hdf5", "w")
        test_dset['ple'] = shuffled_te_ple
        test_dset['abp'] = shuffled_te_abp
        test_dset['size'] = shuffled_te_size
        print('test data for BPNet saved')

        if cv >= 1:
            print('cross validation :', cv)
            k_fold_cv(model_name, save_path, [shuffled_tr_ple, shuffled_tr_abp, shuffled_tr_size], cv)
            print('train data for BPNet saved')

        else:
            print('not supported fold number')
    elif model_name == "Unet":
        total_data = [[p, a] for p, a in zip(ple, abp)]
        random.shuffle(total_data)
        total_len = len(total_data)
        train_val_len = int(total_len * 0.8)
        shuffled_ple = np.squeeze([n[0] for n in total_data])
        shuffled_abp = np.squeeze([n[1] for n in total_data])
        print('***********\nnp.shape(shuffled_ple) :', np.shape(shuffled_ple))
        shuffled_tr_ple, shuffled_te_ple = np.split(shuffled_ple, [train_val_len])
        shuffled_tr_abp, shuffled_te_abp = np.split(shuffled_abp, [train_val_len])
        print('np.shape(shuffled_tr_ple) :', np.shape(shuffled_tr_ple))
        print('np.shape(shuffled_te_ple) :', np.shape(shuffled_te_ple))

        print('data shuffle for Unet done')
        test_dset = h5py.File(save_path + "_test.hdf5", "w")
        test_dset['ple'] = shuffled_te_ple
        test_dset['abp'] = shuffled_te_abp
        print('test data for Unet saved')

        if cv >= 1:
            print('cross validation :', cv)
            k_fold_cv(model_name, save_path, [shuffled_tr_ple, shuffled_tr_abp], cv)
            print('train data for Unet saved')

        else:
            print('not supported fold number')


def k_fold_cv(model_name, save_path, train_data, cv_n):
    if model_name == "BPNet":
        ple = train_data[0]
        abp = train_data[1]
        size = train_data[2]

        if len(ple) % cv_n != 0:
            ple = ple[:len(ple) - len(ple) % cv_n]
            abp = abp[:len(abp) - len(abp) % cv_n]
            size = size[:len(size) - len(size) % cv_n]

        ple_total, abp_total, size_total = [], [], []
        ple_validation_total, abp_validation_total, size_validation_total = [], [], []

        if cv_n == 1:
            # normal train validation >> divide train : 80% / validation : 20%
            # ple_total, abp_total, size_total = ple, abp, size
            ple_total, ple_validation_total = np.split(ple, [int(len(ple) * 0.8)])
            abp_total, abp_validation_total = np.split(abp, [int(len(abp) * 0.8)])
            size_total, size_validation_total = np.split(size, [int(len(size) * 0.8)])

        else:
            # n fold cross validation >> divide data into n folds
            ple_folds = np.split(ple, cv_n)
            abp_folds = np.split(abp, cv_n)
            size_folds = np.split(size, cv_n)
            for idx, (ple_fold, abp_fold, size_fold) in enumerate(zip(ple_folds, abp_folds, size_folds)):
                ple_temp = []
                abp_temp = []
                size_temp = []
                for i in range(cv_n):
                    if idx != i:
                        ple_temp.append(ple_folds[i])
                        abp_temp.append(abp_folds[i])
                        size_temp.append(size_folds[i])
                        # print(i)
                ple_total.append(np.reshape(ple_temp, (-1, 3, chunk_size)))
                abp_total.append(np.reshape(abp_temp, (-1, chunk_size)))
                size_total.append(np.reshape(size_temp, (-1, 3)))

                ple_validation_total.append(ple_fold)
                abp_validation_total.append(abp_fold)
                size_validation_total.append(size_fold)

        dset = h5py.File(save_path + '_train(cv' + str(cv_n) + ').hdf5', "w")
        t_dset = dset.create_group('train')
        t_dset.attrs['desc'] = "k-fold cv train dataset"
        t_p = t_dset.create_group('ple')
        t_a = t_dset.create_group('abp')
        t_s = t_dset.create_group('size')

        v_dset = dset.create_group('validation')
        v_dset.attrs['desc'] = "k-fold cv validation dataset"
        v_p = v_dset.create_group('ple')
        v_a = v_dset.create_group('abp')
        v_s = v_dset.create_group('size')
        if cv_n == 1:
            t_p.create_dataset(name=str(0), data=ple_total)
            t_a.create_dataset(name=str(0), data=abp_total)
            t_s.create_dataset(name=str(0), data=size_total)
            v_p.create_dataset(name=str(0), data=ple_validation_total)
            v_a.create_dataset(name=str(0), data=abp_validation_total)
            v_s.create_dataset(name=str(0), data=size_validation_total)
        else:
            for n in range(cv_n):
                t_p.create_dataset(name=str(n), data=ple_total[n])
                t_a.create_dataset(name=str(n), data=abp_total[n])
                t_s.create_dataset(name=str(n), data=size_total[n])
                v_p.create_dataset(name=str(n), data=ple_validation_total[n])
                v_a.create_dataset(name=str(n), data=abp_validation_total[n])
                v_s.create_dataset(name=str(n), data=size_validation_total[n])

        print('t_dset.keys() :', t_dset.keys())
        print('t_p.keys() :', t_p.keys())
        print('done')
        hdf_handler.cv_dataset_reader(save_path + '_train(cv' + str(cv_n) + ').hdf5')
    elif model_name == "Unet":
        ple = train_data[0]
        abp = train_data[1]
        if len(ple) % cv_n != 0:
            ple = ple[:len(ple) - len(ple) % cv_n]
            abp = abp[:len(abp) - len(abp) % cv_n]

        if cv_n == 1:
            ple_total, ple_validation_total = np.split(ple, [int(len(ple) * 0.8)])
            abp_total, abp_validation_total = np.split(abp, [int(len(abp) * 0.8)])

        else:  # cv_n > 1
            ple_folds = np.split(ple, cv_n)
            abp_folds = np.split(abp, cv_n)
            ple_total, abp_total = [], []
            ple_validation_total, abp_validation_total = [], []

            for idx, (ple_fold, abp_fold) in enumerate(zip(ple_folds, abp_folds)):
                ple_temp = []
                abp_temp = []
                for i in range(cv_n):
                    if idx != i:
                        ple_temp.append(ple_folds[i])
                        abp_temp.append(abp_folds[i])
                        # print(i)
                ple_total.append(np.reshape(ple_temp, (-1, chunk_size)))
                abp_total.append(np.reshape(abp_temp, (-1, chunk_size)))

                ple_validation_total.append(ple_fold)
                abp_validation_total.append(abp_fold)

        dset = h5py.File(save_path + '_train(cv' + str(cv_n) + ').hdf5', "w")
        t_dset = dset.create_group('train')
        t_dset.attrs['desc'] = "k-fold cv train dataset"
        t_p = t_dset.create_group('ple')
        t_a = t_dset.create_group('abp')

        v_dset = dset.create_group('validation')
        v_dset.attrs['desc'] = "k-fold cv validation dataset"
        v_p = v_dset.create_group('ple')
        v_a = v_dset.create_group('abp')

        for n in range(cv_n):
            t_p.create_dataset(name=str(n), data=ple_total[n])
            t_a.create_dataset(name=str(n), data=abp_total[n])
            v_p.create_dataset(name=str(n), data=ple_validation_total[n])
            v_a.create_dataset(name=str(n), data=abp_validation_total[n])

        print('t_dset.keys() :', t_dset.keys())
        print('t_p.keys() :', t_p.keys())
        print('done')
        # hdf_handler.cv_dataset_reader(save_path + '_train(cv' + str(cv_n) + ').hdf5')
    else:
        print('not supported model in cv_dataset_maker')
        return
