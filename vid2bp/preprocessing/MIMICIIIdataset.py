import os
import wfdb

from tqdm import tqdm
# import logging
# import logging.handlers
import multiprocessing as mp
import numpy as np
import random


# log = logging.getLogger('MIMICIII')
# log.setLevel(logging.DEBUG)
# log.addHandler(logging.handlers.RotatingFileHandler(filename='MIMICIII_log.txt', mode='w'))


# test_path = '/hdd/hdd0/dataset/bpnet/physionet.org/files/mimic3wdb/1.0/30/3000866/3000866'
# record = wfdb.rdrecord(test_path)
# print(record.sig_name)

def get_processor_num(target_num: int):
    divisors = []
    for i in range(1, int(target_num ** 0.5) + 1):
        if target_num % i == 0:
            divisors.append(i)
            if i != target_num // i:
                divisors.append(target_num // i)
    test = [x for x in divisors if x < os.cpu_count()]
    if np.max(test) < os.cpu_count() // 2:
        return os.cpu_count()
    else:
        return np.max(test)


def remove_extension(path: str):
    if path.endswith('.hea'):
        return path[:-4]
    else:
        return path


def read_record_per_patient(segments: list):
    cnt: int = 0
    for s in segments:
        if os.path.isfile(s) and os.path.isfile(s.replace('.hea', '.dat')):
            record = wfdb.rdrecord(remove_extension(s))
            if ('PLETH' in record.sig_name) and ('ABP' in record.sig_name):
                cnt += 1
            else:
                file_remove_command = 'rm -rf ' + s + ' ' + s.replace('.hea', '.dat')
                os.system(file_remove_command)
                continue
        else:
            file_remove_command = 'rm -rf ' + s.replace('.hea', '.*')
            os.system(file_remove_command)

    if cnt == 0:
        # print(path[0].split('/')[-2], 'Neither ABP nor PLETH found removed patient')
        # if any of the record is not available, remove the whole patient
        try:
            directory_remove_command = 'rm -rf ' + segments[0][:-17]
            os.system(directory_remove_command)
        except:
            directory_rename_command = 'mv ' + segments[0][:-17] + ' ' + segments[0][:-17] + '_' + str(cnt)
            os.system(directory_rename_command)
            pass
    else:
        directory_rename_command = 'mv ' + segments[0][:-17] + ' ' + segments[0][:-17] + '_' + str(cnt)
        os.system(directory_rename_command)


def get_patients_list(record_path: str, is_subset: bool = True, is_neonates: bool = True):
    """
    save_path = '/hdd/hdd0/dataset/bpnet/neonates/'
    """
    if is_subset:
        read_path = record_path + 'RECORDS'
    else:
        read_path = record_path + 'RECORDS-neonates' if is_neonates else record_path + 'RECORDS-adults'

    with open(read_path, 'r') as f:
        records = f.readlines()

    return [record.strip('\n') for record in records]


def download_patients(is_subset: bool, save_path: str, patients: list):
    '''
    param:
        save_path: path to save the data (e.g. /hdd/hdd0/dataset/neonates/)
        patients: list of patients to check
    '''
    # save_path = '/hdd/hdd0/dataset/bpnet/neonates/'
    if is_subset:
        download_root_url = 'https://physionet.org/files/mimic3wdb-matched/1.0/'
        mimiciii_internal_path = 'physionet.org/files/mimic3wdb-matched/1.0/'
    else:
        download_root_url = 'https://physionet.org/files/mimic3wdb/1.0/'
        mimiciii_internal_path = 'physionet.org/files/mimic3wdb/1.0/'

    pbar = tqdm(patients, total=len(patients), position=0, ncols=70, ascii=' =',
                leave=True)

    for patient in pbar:
        pbar.set_description(f'Patient: {patient}')
        # download a patient data before reading with wget
        terminal_command = 'wget -r -N -c -np -P ' + save_path + ' ' + download_root_url + patient + ' -q'
        # if progress bar needed, add : " --show-progress --progress=bar:force" to terminal_command
        os.system(terminal_command)

        # get a list of segments of a patient
        segments = os.listdir(save_path + mimiciii_internal_path + patient)
        segments_dir = [save_path + mimiciii_internal_path + patient + s for s in segments if
                        (s.endswith('.hea') and ('_' in s) and ('layout' not in s))]
        read_record_per_patient(segments_dir)
    pbar.close()


def split_patients(patients: list, num_of_split: int):
    '''
    **used if multi wget is needed
    param:
        patients: list of patients
        num_of_split: number of split
    return:
        a list of list of patients
    '''
    # patients = get_patients_list(save_path, is_neonates)
    split_size = len(patients) // num_of_split
    patients_split = [patients[i:i + split_size] for i in range(0, len(patients), split_size)]
    return patients_split


def multi_wget(is_subset: bool, is_neonates: bool):
    '''
    param
        is_neonates: True if were to download neonates data, else False
    return:
        None
    '''
    # save_path = '/hdd/hdd0/dataset/bpnet/neonates/'
    if is_subset:
        save_path = '/hdd/hdd1/dataset/mimiciiisubset/'
        # save_path = save_path
    else:
        save_path = '/hdd/hdd1/dataset/bpnet/'
        if is_neonates:
            save_path = save_path + 'neonates/'
        else:
            save_path = save_path + 'adults/'
    read_path = save_path + 'records/'

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    patients = get_patients_list(record_path=read_path, is_subset=is_subset, is_neonates=is_neonates)
    random.shuffle(patients)

    total_len = len(patients)
    processor_num = get_processor_num(total_len)
    process_per_processor = total_len // processor_num
    res = total_len % processor_num
    process = []

    for i in range(processor_num):
        chunk = patients[i * process_per_processor: (i + 1) * process_per_processor]
        proc = mp.Process(target=download_patients, args=(is_subset, save_path, chunk))
        process.append(proc)
        proc.start()
    for proc in process:
        proc.join()

    if res != 0:
        chunk = patients[-res:]
        download_patients(is_subset, save_path, chunk)


'''
subset: 1.0 data takes 2.5 days to download
neonates : takes about 2 days to download
'''

# multi_wget(is_subset=True, is_neonates=False)  # ,split_index=sys.argv[1])
multi_wget(is_subset=False, is_neonates=True)  # ,split_index=sys.argv[1])
# multi_wget(is_subset=False, is_neonates=False)  # ,split_index=sys.argv[1])
