import numpy as np
import mat73
import os
import cnibp.preprocessing.utils.signal_utils as su


def data_aggregator(model_name, root_path, chunk_size=750, samp_rate=60):
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
        for l in range(int(len(ppg) / chunk_size)):  # default chunk size is 1024 (8.192sec) at sampling rate of 125Hz
            ppg_temp = ppg[l * chunk_size:(l + 1) * chunk_size]
            abp_temp = abp[l * chunk_size:(l + 1) * chunk_size]
            ppg_total.append(ppg_temp)
            abp_total.append(abp_temp)

    ppg_total = np.expand_dims(ppg_total, axis=1)[:100]
    abp_total = np.expand_dims(abp_total, axis=1)[:100]
    sig_total = np.swapaxes(np.concatenate([abp_total, ppg_total], axis=1), 1, 2)[:100]
    abp, ple, dsm = su.signal_slicing(model_name, sig_total, chunk_size, samp_rate, fft=True)

    return ple, abp, dsm
