import cv2
import os
import glob
import numpy as np
import config as config
from tqdm import tqdm
import matplotlib.pyplot as plt
# used for accessing url to download files
import urllib.request as urlreq
from sklearn import preprocessing
from joblib import Parallel, delayed, parallel_backend
import time

# download requisite certificates
import ssl;

ssl._create_default_https_context = ssl._create_stdlib_context


# Chunks the ROI into blocks of size 5x5
def chunkify(img, block_width=5, block_height=5):
    shape = img.shape
    x_len = shape[0] // block_width
    y_len = shape[1] // block_height
    # print(x_len, y_len)

    chunks = []
    x_indices = [i for i in range(0, shape[0] + 1, x_len)]
    y_indices = [i for i in range(0, shape[1] + 1, y_len)]

    shapes = list(zip(x_indices, y_indices))

    #  # for plotting purpose
    # implot = plt.imshow(img)
    #
    # end_x_list = []
    # end_y_list = []


    for i in range(len(x_indices) - 1):
        # try:
        start_x = x_indices[i]
        end_x = x_indices[i + 1]
        for j in range(len(y_indices) - 1):
            start_y = y_indices[j]
            end_y = y_indices[j+1]
            # end_x_list.append(end_x)
            # end_y_list.append(end_y)
            chunks.append(img[start_x:end_x, start_y:end_y])
        # except IndexError:
        #     print('End of Array')

    return chunks


def plot_image(img):
    plt.axis("off")
    plt.imshow(img, origin='upper')
    plt.show()


# Downloads xml file for face detection cascade
def get_haarcascade():
    haarcascade_url = config.haarcascade_url
    haarcascade_filename = haarcascade_url.split('/')[-1]
    # chech if file is in working directory
    if haarcascade_filename in os.listdir(os.curdir):
        # print("xml file already exists")
        pass
    else:
        # download file from url and save locally as haarcascade_frontalface_alt2.xml, < 1MB
        urlreq.urlretrieve(haarcascade_url, haarcascade_filename)
        print("xml file downloaded")

    return cv2.CascadeClassifier(haarcascade_filename)

# Downloads xml file for eye detection cascade
def get_eye_cascade():
    eye_cascade_url = config.eye_cascade_url
    eye_cascade_filename = eye_cascade_url.split('/')[-1]
    # chech if file is in working directory
    if eye_cascade_filename in os.listdir(os.curdir):
        # print("xml file already exists")
        pass
    else:
        # download file from url and save locally as haarcascade_frontalface_alt2.xml, < 1MB
        urlreq.urlretrieve(eye_cascade_url, eye_cascade_filename)
        print("xml file downloaded")

    return cv2.CascadeClassifier(eye_cascade_filename)


# Function to read the the video data as an array of frames and additionally return metadata like FPS, Dims etc.
def get_frames_and_video_meta_data(video_path, meta_data_only=False):
    cap = cv2.VideoCapture(video_path)
    frameRate = cap.get(5)  # frame rate

    # Frame dimensions: WxH
    frame_dims = (int(cap.get(3)), int(cap.get(4)))
    # Paper mentions a stride of 0.5 seconds = 15 frames
    sliding_window_stride = int(frameRate / 2)
    num_frames = int(cap.get(7))
    if meta_data_only:
        return {"frame_rate": frameRate, "sliding_window_stride": sliding_window_stride, "num_frames": num_frames}

    # Frames from the video have shape NumFrames x H x W x C
    frames = np.zeros((num_frames, frame_dims[1], frame_dims[0], 3), dtype='uint8')

    frame_counter = 0
    while cap.isOpened():
        # curr_frame_id = int(cap.get(1))  # current frame number
        ret, frame = cap.read()
        if not ret:
            break

        frames[frame_counter, :, :, :] = frame
        frame_counter += 1
        if frame_counter == num_frames:
            break

    cap.release()
    return frames, frameRate, sliding_window_stride


# Threaded function for st_map generation from a single video arg:file in dataset
def get_spatio_temporal_map_threaded(file):
    # print(f"Generating Maps for file: {file}")
    # maps = np.zeros((10, config.CLIP_SIZE, 25, 3))
        # print(index)
    maps = preprocess_video_to_st_maps(
        video_path=file,
        output_shape=(180, 180), clip_size=config.CLIP_SIZE)

    if maps is None:
        return 1

    file_name = file.split('/')[-1].split('.')[0]
    folder_name = file.split('/')[-2]
    save_path = os.path.join(config.ST_MAPS_PATH, folder_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, f"{file_name}.npy")
    # np.save(f"{config.ST_MAPS_PATH}{file_name}.npy", maps)
    np.save(save_path, maps)
    return 1


# Threaded wrapper function for st_maps from all videos that calls the threaded func in a parallel fashion
def get_spatio_temporal_map_threaded_wrapper():
    video_files = glob.glob(config.FACE_DATA_DIR + '*avi')
    # video_files = video_files[:10]
    less_than_ten = ['/Volumes/T7/vipl_videos/p19_v2_source2.avi', '/Volumes/T7/vipl_videos/p32_v7_source3.avi', '/Volumes/T7/vipl_videos/p32_v7_source4.avi', '/Volumes/T7/vipl_videos/p40_v7_source2.avi', '/Volumes/T7/vipl_videos/p22_v3_source1.avi']
    video_files = [file for file in video_files if file not in less_than_ten]
    start = time.time()
    with parallel_backend("loky", inner_max_num_threads=4):
        Parallel(n_jobs=3)(delayed(get_spatio_temporal_map_threaded)(file) for file in tqdm(video_files))
    end = time.time()

    print('{:.4f} s'.format(end - start))


# function for st_map generation from all videos in dataset
def get_spatio_temporal_map():
    video_files = glob.glob(config.FACE_DATA_DIR + '*avi')
    # video_files = video_files[100:110]
    # video_files = ['/Volumes/Backup Plus/vision/vipl_videos/p10_v1_source1.avi', '/Volumes/Backup Plus/vision/vipl_videos/p10_v1_source2.avi']
    # video_files = ['/Volumes/Backup Plus/vision/DEAP_emotion/face_video/s01/s01_trial01.avi']
    start = time.time()
    for file in tqdm(video_files):
        # maps = np.zeros((1, config.CLIP_SIZE, 25, 3))
        # for index in range(1):
            # print(index)
        if os.path.exists(f"{config.ST_MAPS_PATH}{file.split('/')[-1].split('.')[0]}.npy"):
            map = np.load(f"{config.ST_MAPS_PATH}{file.split('/')[-1].split('.')[0]}.npy")
            if (map.shape[0]) > 125:
                print(f"\nFilename:{file} | num maps: {map.shape[0]}")
        else:
            continue
        # maps = preprocess_video_to_st_maps(
        #     video_path=file,
        #     output_shape=(180, 180), clip_size=config.CLIP_SIZE)
        # if maps is None:
        #     continue
        optimized_end = time.time()
        # print('{:.4f} s'.format((optimized_end - start)/60))

        # file_name = file.split('/')[-1].split('.')[0]
        # folder_name = file.split('/')[-2]
        # save_path = os.path.join(config.ST_MAPS_PATH, folder_name)
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        # save_path = os.path.join(save_path, f"{file_name}.npy")
        # # np.save(f"{config.ST_MAPS_PATH}{file_name}.npy", maps)
        # np.save(save_path, maps)

    end = time.time()
    print('{:.4f} s'.format(end - start))
    # return maps


# Optimized function for converting videos to Spatio-temporal maps
def preprocess_video_to_st_maps(video_path, output_shape, clip_size):
    frames, frameRate, sliding_window_stride = get_frames_and_video_meta_data(video_path)

    num_frames = frames.shape[0]
    output_shape = (frames.shape[1], frames.shape[2])
    num_maps = int((num_frames - clip_size)/sliding_window_stride + 1)
    if num_maps < 0:
        # print(num_maps)
        print(video_path)
        return None

    # stacked_maps is the all the st maps for a given video (=num_maps) stacked.
    stacked_maps = np.zeros((num_maps, config.CLIP_SIZE, 25, 3))
    # processed_maps will contain all the data after processing each frame, but not yet converted into maps
    processed_maps = np.zeros((num_frames, 25, 3))
    # processed_frames = np.zeros((num_frames, output_shape[0], output_shape[1], 3))
    processed_frames = []
    map_index = 0

    # Init scaler and detector
    min_max_scaler = preprocessing.MinMaxScaler()
    detector = get_haarcascade()
    eye_detector = get_eye_cascade()

    # First we process all the frames and then work with sliding window to save repeated processing for the same frame index
    for idx, frame in enumerate(frames):
        # spatio_temporal_map = np.zeros((fr, 25, 3))
        '''
           Preprocess the Image
           Step 1: Use cv2 face detector based on Haar cascades
           Step 2: Crop the frame based on the face co-ordinates (we need to do 160%)
           Step 3: Downsample the face cropped frame to output_shape = 36x36
       '''
        faces = detector.detectMultiScale(frame, 1.3, 5)
        if len(faces) is not 0:
            (x, y, w, d) = faces[0]
            frame_cropped = frame[y:(y + d), x:(x + w)]
            eyes = eye_detector.detectMultiScale(frame_cropped, 1.2, 3)
            # if len(eyes) > 0:
            #     # for having the same radius in both eyes
            #     (eye_x, eye_y, eye_w, eye_h) = eyes[0]
            #     eye_radius = (eye_w + eye_h) // 5
            #     mask = np.ones(frame_cropped.shape[:2], dtype="uint8")
            #     for (ex, ey, ew, eh) in eyes[:2]:
            #         eye_center = (ex + ew // 2, ey + eh // 2)
            #         # if eye_radius
            #         cv2.circle(mask, eye_center, eye_radius, 0, -1)
            #         # eh = int(0.8*eh)
            #         # ew = int(0.8*ew)
            #         # cv2.rectangle(mask, (ex, ey), (ex+ew, ey+eh), 0, -1)
            #
            #     frame_masked = cv2.bitwise_and(frame_cropped, frame_cropped, mask=mask)
            # else:
            #     frame_masked = frame_cropped
            #     # plot_image(frame_masked)

            frame_masked = frame_cropped
        else:
            # The problemis that this doesn't get cropped :/
            # (x, y, w, d) = (308, 189, 215, 215)
            # frame_masked = frame[y:(y + d), x:(x + w)]

            # print("face detection failed, image frame will be masked")
            mask = np.zeros(frame.shape[:2], dtype="uint8")
            frame_masked = cv2.bitwise_and(frame, frame, mask=mask)
            # plot_image(frame_masked)

        # frame_cropped = frame[y:(y + d), x:(x + w)]

        try:
            # frame_resized = cv2.resize(frame_masked, output_shape, interpolation=cv2.INTER_CUBIC)
            frame_resized = cv2.cvtColor(frame_masked, cv2.COLOR_BGR2YUV)

        except:
            print('\n--------- ERROR! -----------\nUsual cv empty error')
            print(f'Shape of img1: {frame.shape}')
            # print(f'bbox: {bbox}')
            print(f'This is at idx: {idx}')
            exit(666)

        processed_frames.append(frame_resized)
        # roi_blocks = chunkify(frame_resized)
        # for block_idx, block in enumerate(roi_blocks):
        #     avg_pixels = cv2.mean(block)
        #     processed_maps[idx, block_idx, 0] = avg_pixels[0]
        #     processed_maps[idx, block_idx, 1] = avg_pixels[1]
        #     processed_maps[idx, block_idx, 2] = avg_pixels[2]

    # At this point we have the processed maps from all the frames in a video and now we do the sliding window part.
    for start_frame_index in range(0, num_frames, sliding_window_stride):
        end_frame_index = start_frame_index + clip_size
        if end_frame_index > num_frames:
            break
        # # print(f"start_idx: {start_frame_index} | end_idx: {end_frame_index}")
        spatio_temporal_map = np.zeros((clip_size, 25, 3))
        #
        # spatio_temporal_map = processed_maps[start_frame_index:end_frame_index, :, :]


        for idx, frame in enumerate(processed_frames[start_frame_index:end_frame_index]):
            roi_blocks = chunkify(frame)
            for block_idx, block in enumerate(roi_blocks):
                avg_pixels = cv2.mean(block)
                spatio_temporal_map[idx, block_idx, 0] = avg_pixels[0]
                spatio_temporal_map[idx, block_idx, 1] = avg_pixels[1]
                spatio_temporal_map[idx, block_idx, 2] = avg_pixels[2]

        for block_idx in range(spatio_temporal_map.shape[1]):
            # Not sure about uint8
            fn_scale_0_255 = lambda x: (x * 255.0).astype(np.uint8)
            scaled_channel_0 = min_max_scaler.fit_transform(spatio_temporal_map[:, block_idx, 0].reshape(-1, 1))
            spatio_temporal_map[:, block_idx, 0] = fn_scale_0_255(scaled_channel_0.flatten())
            scaled_channel_1 = min_max_scaler.fit_transform(spatio_temporal_map[:, block_idx, 1].reshape(-1, 1))
            spatio_temporal_map[:, block_idx, 1] = fn_scale_0_255(scaled_channel_1.flatten())
            scaled_channel_2 = min_max_scaler.fit_transform(spatio_temporal_map[:, block_idx, 2].reshape(-1, 1))
            spatio_temporal_map[:, block_idx, 2] = fn_scale_0_255(scaled_channel_2.flatten())

        stacked_maps[map_index, :, :, :] = spatio_temporal_map
        map_index += 1

    return stacked_maps

# UNOPTIMIZED CODE
# def get_st_maps(video_path, output_shape, clip_size):
#     frames, frameRate, sliding_window_stride = get_frames_and_video_meta_data(video_path)
#
#     num_frames = frames.shape[0]
#     num_maps = int((num_frames - clip_size)/sliding_window_stride + 1)
#     maps = np.zeros((num_maps, config.CLIP_SIZE, 25, 3))
#     map_index = 0
#
#     # Init scaler and detector
#     min_max_scaler = preprocessing.MinMaxScaler()
#     detector = get_haarcascade()
#     eye_detector = get_eye_cascade()
#
#     for start_frame_index in tqdm(range(0, num_frames, sliding_window_stride)):
#         end_frame_index = start_frame_index + clip_size
#         if end_frame_index > 400:
#             break
#         # print(f"start_idx: {start_frame_index} | end_idx: {end_frame_index}")
#         spatio_temporal_map = np.zeros((clip_size, 25, 3))
#
#         frames_in_clip = frames[start_frame_index:end_frame_index]
#
#         for idx, frame in enumerate(frames_in_clip):
#             '''
#                Preprocess the Image
#                Step 1: Use cv2 face detector based on Haar cascades
#                Step 2: Crop the frame based on the face co-ordinates (we need to do 160%)
#                Step 3: Downsample the face cropped frame to output_shape = 36x36
#            '''
#             faces = detector.detectMultiScale(frame, 1.3, 5)
#             if len(faces) is not 0:
#                 (x, y, w, d) = faces[0]
#                 frame_cropped = frame[y:(y + d), x:(x + w)]
#                 eyes = eye_detector.detectMultiScale(frame_cropped, 1.2, 3)
#                 if len(eyes) > 0:
#                     # for having the same radius in both eyes
#                     (eye_x, eye_y, eye_w, eye_h) = eyes[0]
#                     eye_radius = (eye_w + eye_h) // 5
#                     mask = np.ones(frame_cropped.shape[:2], dtype="uint8")
#                     for (ex, ey, ew, eh) in eyes[:2]:
#                         eye_center = (ex + ew // 2, ey + eh // 2)
#                         # if eye_radius
#                         cv2.circle(mask, eye_center, eye_radius, 0, -1)
#                         # eh = int(0.8*eh)
#                         # ew = int(0.8*ew)
#                         # cv2.rectangle(mask, (ex, ey), (ex+ew, ey+eh), 0, -1)
#
#                     frame_masked = cv2.bitwise_and(frame_cropped, frame_cropped, mask=mask)
#                 else:
#                     frame_masked = frame_cropped
#                     # plot_image(frame_masked)
#             else:
#                 # The problemis that this doesn't get cropped :/
#                 # (x, y, w, d) = (308, 189, 215, 215)
#                 # frame_masked = frame[y:(y + d), x:(x + w)]
#
#                 # print("face detection failed, image frame will be masked")
#                 mask = np.zeros(frame.shape[:2], dtype="uint8")
#                 frame_masked = cv2.bitwise_and(frame, frame, mask=mask)
#                 # plot_image(frame_masked)
#
#             # frame_cropped = frame[y:(y + d), x:(x + w)]
#
#             try:
#                 frame_resized = cv2.resize(frame_masked, output_shape, interpolation=cv2.INTER_CUBIC)
#                 frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2YUV)
#
#             except:
#                 print('\n--------- ERROR! -----------\nUsual cv empty error')
#                 print(f'Shape of img1: {frame.shape}')
#                 # print(f'bbox: {bbox}')
#                 print(f'This is at idx: {idx}')
#                 exit(666)
#
#             roi_blocks = chunkify(frame_resized)
#             for block_idx, block in enumerate(roi_blocks):
#                 avg_pixels = cv2.mean(block)
#                 spatio_temporal_map[idx, block_idx, 0] = avg_pixels[0]
#                 spatio_temporal_map[idx, block_idx, 1] = avg_pixels[1]
#                 spatio_temporal_map[idx, block_idx, 2] = avg_pixels[2]
#
#         print('he;;p')
#
#         for block_idx in range(spatio_temporal_map.shape[1]):
#             # Not sure about uint8
#             fn_scale_0_255 = lambda x: (x * 255.0)
#             scaled_channel_0 = min_max_scaler.fit_transform(spatio_temporal_map[:, block_idx, 0].reshape(-1, 1))
#             spatio_temporal_map[:, block_idx, 0] = fn_scale_0_255(scaled_channel_0.flatten())
#             scaled_channel_1 = min_max_scaler.fit_transform(spatio_temporal_map[:, block_idx, 1].reshape(-1, 1))
#             spatio_temporal_map[:, block_idx, 1] = fn_scale_0_255(scaled_channel_1.flatten())
#             scaled_channel_2 = min_max_scaler.fit_transform(spatio_temporal_map[:, block_idx, 2].reshape(-1, 1))
#             spatio_temporal_map[:, block_idx, 2] = fn_scale_0_255(scaled_channel_2.flatten())
#
#         maps[map_index, :, :, :] = spatio_temporal_map
#         map_index += 1
#
#     return maps


if __name__ == '__main__':
    # get_frames_and_video_meta_data('/Volumes/T7/vipl_videos/p58_v4_source3.avi')
    # get_spatio_temporal_map()
    # get_spatio_temporal_map_threaded_wrapper()
    # video_files = glob.glob(config.FACE_DATA_DIR + '/**/*avi')
    # r = list(process_map(get_spatio_temporal_map_threaded, video_files[:2], max_workers=1))
    # signal = read_target_data("/Users/anweshcr7/github/RhythmNet/data/data_preprocessed/", "s01_trial04")
    #
    # resampled = signal.resample(df["Signal"].values, 3000, df["Time"].values)
    # resampled_sample_rate = hp.get_samplerate_mstimer(resampled[1])
    # print(calculate_hr(resampled[0], resampled_sample_rate))

    # make_csv_with_frame_rate()
    print('done')