import os
import glob
import numpy as np
import pandas as pd
import torch
from scipy import signal
import heartpy as hp
from tqdm import tqdm
import config as config
import utils.video2st_maps as video2st_maps
import matplotlib.pyplot as plt


# To be used for DEAP dataset where the PPG signal is data[38]
def get_ppg_channel(x):
    # i think PPG channel is at 38
    return x[38]


# Reads the clip-wise HR data that was computed and stored in the csv files (per video)
def get_hr_data(file_name):
    hr_df = pd.read_csv(config.HR_DATA_PATH + f"{file_name}.csv")

    return hr_df["hr_bpm"].values


# Read the raw signal from the ground truth csv and resample.
# Not be needed during the model as we will compute the HRs first-hand and use them directly instead of raw signals
def read_target_data(target_data_path, video_file_name):
    signal_data_file_path = os.path.join(target_data_path, f"{video_file_name} PPG.csv")
    signal_df = pd.read_csv(signal_data_file_path)

    return signal_df["Signal"].values, signal_df["Time"].values
    # In RhythmNet maybe we don't need to resample. CHECK
    return filter_and_resample_truth_signal(signal_df, resampling_size=3000)


# Function allows filtering and resampling of signals. Not being used for VIPL-HR
def filter_and_resample_truth_signal(signal_df, resampling_size):
    # Signal should be bandpass filtered to remove noise outside of expected HR frequency range.
    # But we are using CLEANER_PPG signals which are considered filtered.
    orignal_sample_rate = hp.get_samplerate_mstimer(signal_df["Time"].values)

    # filtered = hp.filter_signal(signal_df["Signal"].values, [0.7, 2.5], sample_rate=sample_rate,
    #                             order=3, filtertype='bandpass')
    resampled_signal = signal.resample(signal_df["Signal"].values, resampling_size, t=signal_df["Time"].values)

    # we'll need to add resampled[1]
    return resampled_signal[0], resampled_signal[1]


# Returns index of value that is nearest to the arg:value in the arg:array
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


# Controller Function to compute and store the HR values as csv (HR values measured clip-wise i.e. per st_map per video)
def compute_hr_for_rhythmnet():
    data_files = glob.glob(config.TARGET_SIGNAL_DIR + "*.csv")
    # for file in tqdm(data_files):
    for file in tqdm(data_files[:1]):
        file = '/Users/anweshcr7/Downloads/CleanerPPG/VIPL-HR/Cleaned/p41_v7_source2.csv'
        signal_df = pd.read_csv(file)
        signal_data, timestamps, peak_data = signal_df["Signal"].values, signal_df["Time"].values, signal_df["Peaks"].values
        video_path = config.FACE_DATA_DIR + f"{file.split('/')[-1].split('.')[0]}.avi"
        video_meta_data = video2st_maps.get_frames_and_video_meta_data(video_path, meta_data_only=True)
        # hr_segmentwise = hp.process_segmentwise(signal_df["Signal"].values, sample_rate=128, segment_width=10, segment_overlap=0.951)
        # hr_segmentwise = hr_segmentwise[1]["bpm"]
        # plt.plot(np.arange(len(hr_segmentwise)), hr_segmentwise)
        # plt.show()
        npy_path = f"{config.ST_MAPS_PATH}{file.split('/')[-1].split('.')[0]}.npy"
        if os.path.exists(npy_path):
            video_meta_data["num_maps"] = np.load(f"{config.ST_MAPS_PATH}{file.split('/')[-1].split('.')[0]}.npy").shape[0]
        else:
            continue
        hr = np.asarray(calculate_hr_clip_wise(timestamps, signal_df, video_meta_data), dtype="float32")
        file_name = file.split("/")[-1].split(".")[0].split(" ")[0]
        hr_df = pd.DataFrame(hr, columns=["hr_bpm"])
        hr_df.to_csv(f"../data/hr_csv/{file_name}.csv", index=False)
        # print("eheee")


# Function to compute and store the HR values as csv (HR values measured clip-wise i.e. per st_map per video)
def calculate_hr_clip_wise(timestamps=None, signal_df=None, video_meta_data=None):

    sliding_window_stride = int((video_meta_data["sliding_window_stride"]/video_meta_data["frame_rate"])*1000)
    sliding_window_size_frame = int((config.CLIP_SIZE/video_meta_data["frame_rate"]))
    # convert to milliseconds
    sliding_window_size = sliding_window_size_frame * 1000
    # num_maps = int((video_meta_data["num_frames"] - config.CLIP_SIZE)/sliding_window_size_frame) + 1
    num_maps = video_meta_data["num_maps"]
    # for i in range(len(timestamps)):
    #     print(timestamps[i+1]-timestamps[i])
    count = 0
    hr_list = []
    for start_time in range(0, int(timestamps[-1]), sliding_window_stride):
        if count == num_maps:
            break
        # start_index = np.where(timestamps == start_time)
        end_time = start_time + sliding_window_size
        # end_index = np.where(timestamps == end_time)
        start_index = np.searchsorted(timestamps, start_time, side='left')
        end_index = np.searchsorted(timestamps, end_time, side='left')

        # start_index = start_index[0][0]
        if end_index == 0:
            end_index = len(timestamps) - 1
            # break

        curr_data = signal_df.iloc[start_index:end_index]
        time_intervals = curr_data[curr_data["Peaks"] == 1]["Time"].values
        ibi_array = [time_intervals[idx + 1] - time_intervals[idx] for idx, time_val in enumerate(time_intervals[:-1])]
        if len(ibi_array) == 0:
            hr_bpm = hr_list[-1]
        else:
            hr_bpm = 1000/np.mean(ibi_array)*60
        hr_list.append(hr_bpm)

        count += 1


    # plt.plot(np.arange(len(hr_list)), hr_list)
    # plt.show()
    return hr_list


# Function to compute HR from raw signal.
def calculate_hr(signal_data, timestamps=None):
    sampling_rate = 47.63
    if timestamps is not None:
        sampling_rate = hp.get_samplerate_mstimer(timestamps)
        try:
            wd, m = hp.process(signal_data, sample_rate=sampling_rate)
            hr_bpm = m["bpm"]
        except:
            hr_bpm = 75.0

        if np.isnan(hr_bpm):
            hr_bpm = 75.0
        return hr_bpm

    else:
        # We are working with predicted HR:
        # need to filter and do other stuff.. lets see
        signal_data = hp.filter_signal(signal_data, cutoff=[0.7, 2.5], sample_rate=sampling_rate, order=6,
                                       filtertype='bandpass')
        try:
            wd, m = hp.process(signal_data, sample_rate=sampling_rate, high_precision=True, clean_rr=True)
            hr_bpm = m["bpm"]
        except:
            print("BadSignal received (could not be filtered) using def HR value = 75bpm")
            hr_bpm = 75.0
        return hr_bpm


if __name__ == '__main__':
    compute_hr_for_rhythmnet()

    files = glob.glob(config.HR_DATA_PATH+"/*.csv")
    for file in files:
        hr = get_hr_data(file.split('/')[-1].split('.')[0])
        if type(hr) == np.object_:
            print(file)
        try:
            torch.tensor(hr, dtype=torch.float)
        except:
            print(file)