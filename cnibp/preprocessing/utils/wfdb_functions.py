import wfdb

def find_channel_idx(path):
    """
    param: path: path of a segment (e.g. /hdd/hdd0/dataset/bpnet/adults/physionet.org/files/mimic3wdb/1.0/30/3001937_11)
    return: idx: index of the ple, abp channel
    """
    record = wfdb.rdrecord(path)
    channel_names = record.sig_name
    ple_idx = [p for p in range(len(channel_names)) if channel_names[p] == 'PLETH'][0]
    abp_idx = [a for a in range(len(channel_names)) if channel_names[a] == 'ABP'][0]

    return ple_idx, abp_idx
