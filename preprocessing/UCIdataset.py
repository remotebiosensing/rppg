import numpy as np
import mat73
import json
import os
import preprocessing.utils.signal_utils as su
import preprocessing.utils.math_module as mm

with open('/home/paperc/PycharmProjects/VBPNet/config/parameter.json') as f:
    json_data = json.load(f)
    param = json_data.get("parameters")

p = '/home/paperc/PycharmProjects/VBPNet/dataset/uci-database/'


def data_aggregator(root_path, degree=0, train=True, percent=0.75):
    # print(root_path)
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
    if degree == 0:
        abp, ple = su.signal_slicing(sig_total)
        print('*** f data aggregation done***')

        return abp, ple, train_len

    elif degree == 3:
        abp, ple = su.signal_slicing(sig_total)
        ple_first = mm.diff_np(ple)  # f'
        ple_total = mm.diff_channels_aggregator(ple, ple_first)
        print('*** f & f\' data aggregation done***')

        return abp, ple_total, train_len

    elif degree == 6:
        abp, ple = su.signal_slicing(sig_total)  # f
        ple_first = mm.diff_np(ple)  # f'
        ple_second = mm.diff_np(ple_first)  # f''
        ple_total = mm.diff_channels_aggregator(ple, ple_first, ple_second)
        print('*** f & f\' & f\'\' data aggregation done***')

        return abp, ple_total, train_len

    else:
        print('derivative not supported... goto data_aggregator_sig_processed()')


# data_aggregator(p, percent=0.5)

# uci_dat = mat73.loadmat("/home/paperc/PycharmProjects/VBPNet/dataset/uci-database/Part_2.mat")['Part_2']
#
# ppg_total = []
# abp_total = []
#
# for u in uci_dat:
#     ppg, abp, _ = u
#
#     for l in range(int(len(ppg) / 750)):
#         ppg_temp = ppg[l * 750:(l + 1) * 750]
#         abp_temp = abp[l * 750:(l + 1) * 750]
#         ppg_total.append(ppg_temp)
#         abp_total.append(abp_temp)
#
# ppg_total = np.expand_dims(ppg_total, axis=1)
# abp_total = np.expand_dims(abp_total, axis=1)
#
# sig_total = np.swapaxes(np.concatenate([abp_total, ppg_total], axis=1), 1, 2)
#
# ta, tp = sig_process(sig_total)
#
# print(ta[0])
