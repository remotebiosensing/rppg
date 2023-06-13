import wfdb
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def find_channel_idx(path):
    """
    param: path: path of a segment (e.g. /hdd/hdd0/dataset/bpnet/adults/physionet.org/files/mimic3wdb/1.0/30/3001937_11)
    return: idx: index of the ple, abp channel
    """
    # target_channel = ['PLETH', 'ABP']
    # channel_idx = []
    record = wfdb.rdrecord(path)
    channel_names = record.sig_name
    # for tc in target_channel:
    #     if tc in channel_names:
    #         channel_idx.append([i for i in range(len(channel_names)) if channel_names[i] == tc][0])

    ple_idx = [p for p in range(len(channel_names)) if channel_names[p] == 'PLETH'][0]
    abp_idx = [a for a in range(len(channel_names)) if channel_names[a] == 'ABP'][0]
    # try:
    #     ecg_idx = [e for e in range(len(channel_names)) if channel_names[e] == 'II'][0]
    # except:
    #     ecg_idx = -1

    return [ple_idx, abp_idx]


def get_channel_segments(path: str, chunk_size):
    channel_idx = find_channel_idx(path)

    # if channel_idx[-1] == -1:
    #     return False, None, None, None
    segments = np.squeeze(np.split(wfdb.rdrecord(path, channels=channel_idx).p_signal, len(channel_idx), axis=1))
    nan_mask = ~(np.isnan(segments).any(axis=0))
    segments = segments[:, nan_mask]
    if np.shape(segments)[-1] < chunk_size:
        return False, None, None
    else:
        return True, segments[0], segments[1]


def get_channel_record(segment_path: str, channel_idx: int):
    if '.hea' in segment_path:
        segment_path = segment_path.replace('.hea', '')

    record = wfdb.rdrecord(segment_path, channels=[channel_idx])
    digital_sig = np.squeeze(record.adc())
    gain = record.adc_gain[0]
    baseline = record.baseline[0]
    # ple_sig_len = ple_record.sig_len
    # ple_analogue_sig = np.squeeze(ple_record.p_signal)
    # ple_init_value = ple_record.init_value[0]
    # ple_adc_zero = ple_record.adc_zero[0]
    return digital_sig, gain, baseline
