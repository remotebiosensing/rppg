import os
import csv
import numpy as np
from tqdm import tqdm

def slice_signal(file, window_size, stride):
    """
    Helper function for slicing the audio file
    by window size and sample rate with [1-stride] percent overlap (default 20%).
    """
    rdr = csv.reader(open(file,'r'))
    data = []
    for i,line in enumerate(rdr):
        data = list(map(float, line))

    hop = int(window_size * stride)
    slices = []
    for end_idx in range(window_size, len(data), hop):
        start_idx = end_idx - window_size
        slice_sig = data[start_idx:end_idx]
        slices.append(slice_sig)
    return slices


def process_and_serialize(data_type, folder, stride = 0.2, window_size = 256):
    
    if data_type == 'train':
        clean_folder = folder + 'infer_result'
        noisy_folder = folder + 'pred_result'
        serialized_folder = folder + 'serialized_train_data'
    else:
        clean_folder = folder + 'infer_result_test'
        noisy_folder = folder + 'pred_result_test'
        serialized_folder = folder + 'serialized_test_data'
    if not os.path.exists(serialized_folder):
        os.makedirs(serialized_folder)

    # walk through the path, slice the audio file, and save the serialized result
    for root, dirs, files in os.walk(clean_folder):
        if len(files) == 0:
            continue
        for filename in tqdm(files, desc='Serialize and down-sample {} csv'.format(data_type)):
            clean_file = os.path.join(clean_folder, filename)
            noisy_file = os.path.join(noisy_folder, filename)
            # slice both clean signal and noisy signal
            clean_sliced = slice_signal(clean_file, window_size, stride)
            noisy_sliced = slice_signal(noisy_file, window_size, stride)
            # serialize - file format goes [original_file]_[slice_number].npy
            # ex) p293_154.wav_5.npy denotes 5th slice of p293_154.wav file
           
            for idx, slice_tuple in enumerate(zip(clean_sliced, noisy_sliced)):
                pair = np.array([slice_tuple[0], slice_tuple[1]])
                np.save(os.path.join(serialized_folder, '{}_{}'.format(filename.split('.')[0], idx)), arr=pair)
            
                # save to compare the results
                with open(folder+'compare/clean/{}_{}.csv'.format(filename.split('.')[0], idx),'w') as f:
                    wr = csv.writer(f)
                    wr.writerow(slice_tuple[0])
                    f.close()
                with open(folder+'compare/noisy/{}_{}.csv'.format(filename.split('.')[0], idx),'w') as f:
                    wr = csv.writer(f)
                    wr.writerow(slice_tuple[1])
                    f.close()

def data_verify(data_type,folder,window_size = 256):
    """
    Verifies the length of each data after pre-process.
    """
    if data_type == 'train':
        serialized_folder = folder + 'serialized_train_data'
    else:
        serialized_folder = folder + 'serialized_test_data'

    for root, dirs, files in os.walk(serialized_folder):
        for filename in tqdm(files, desc='Verify serialized {} csv'.format(data_type)):
            data_pair = np.load(os.path.join(root, filename))
            if data_pair.shape[1] != window_size:
                print('Snippet length not {} : {} instead'.format(window_size, data_pair.shape[1]))
                break


if __name__ == '__main__':
    process_and_serialize('train')
    data_verify('train')
    process_and_serialize('test')
    data_verify('test')
