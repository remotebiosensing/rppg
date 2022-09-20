import numpy as np
import mat73
import os
import preprocessing.utils.signal_utils as su
import preprocessing.utils.math_module as mm


def data_aggregator(root_path, degree=0, train=True, percent=0.75, samp_rate=60):
    # print(root_path)
    print("reading dataset..")
    file_list = sorted([data for data in os.listdir(root_path) if data.__contains__("Part")])
    d_len = int(len(file_list) * percent)

    if train is True:
        used_list = file_list[:d_len]
    else:
        used_list = file_list[d_len:]
    train_len = len(used_list)
    uci_dat = []
    ppg_total = []
    abp_total = []
    for u in used_list:
        uci_temp = mat73.loadmat(root_path + u)[u.split('.')[0]]
        uci_dat += uci_temp
    print(len(uci_dat))

    for u in uci_dat:
        ppg, abp, _ = u

        for l in range(int(len(ppg) / 750)):
            ppg_temp = ppg[l * 750:(l + 1) * 750]
            abp_temp = abp[l * 750:(l + 1) * 750]
            ppg_total.append(ppg_temp)
            abp_total.append(abp_temp)

    ppg_total = np.expand_dims(ppg_total, axis=1)
    abp_total = np.expand_dims(abp_total, axis=1)

    sig_total = np.swapaxes(np.concatenate([abp_total, ppg_total], axis=1), 1, 2)
    abp, ple, s_list = su.signal_slicing(sig_total, samp_rate, fft=True)

    if degree in [0, 3, 6]:
        if degree == 0:
            ple_total = ple  # f
            print('*** f data aggregation done***')

        elif degree == 3:
            ple_first = mm.diff_np(ple)  # f'
            ple_total = mm.diff_channels_aggregator(ple, ple_first)
            print('*** f & f\' data aggregation done***')

        else:
            ple_first = mm.diff_np(ple)  # f'
            ple_second = mm.diff_np(ple_first)  # f''
            ple_total = mm.diff_channels_aggregator(ple, ple_first, ple_second)
            print('*** f & f\' & f\'\' data aggregation done***')

        return abp, ple_total, s_list, train_len

    else:
        print('derivative not supported... goto data_aggregator_sig_processed()')
