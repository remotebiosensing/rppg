import numpy as np
import mat73
import os
import vid2bp.preprocessing.utils.signal_utils as su
import vid2bp.preprocessing.utils.math_module as mm
import vid2bp.preprocessing.utils.train_utils as tu

import json

def data_aggregator(model_name, root_path, save_path, degree=6, samp_rate=60, cv=1):
    print("save_path : ", save_path)
    print("data_aggregator()\nreading dataset..")
    file_list = sorted([data for data in os.listdir(root_path) if data.__contains__("Part")])

    uci_dat = []
    ppg_total = []
    abp_total = []
    for u in file_list:
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

    if model_name is "VBPNet":
        ppg_total = np.expand_dims(ppg_total, axis=1)
        abp_total = np.expand_dims(abp_total, axis=1)
        sig_total = np.swapaxes(np.concatenate([abp_total, ppg_total], axis=1), 1, 2)
        abp, ple, dsm = su.signal_slicing(model_name, sig_total, samp_rate, fft=True)

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

            tu.data_shuffler(model_name, save_path, [ple_total, abp, dsm], cv)

        else:
            print('derivative not supported... goto data_aggregator_sig_processed()')
    elif model_name is "Unet":
        ppg_total = np.expand_dims(ppg_total, axis=1)
        abp_total = np.expand_dims(abp_total, axis=1)
        sig_total = np.swapaxes(np.concatenate([abp_total, ppg_total], axis=1), 1, 2)
        abp, ple = su.signal_slicing(model_name, sig_total, samp_rate, fft=True)

        tu.data_shuffler(model_name, save_path, [ple, abp], cv)
    else:
        print("model name not supported in data_aggregator2()")
