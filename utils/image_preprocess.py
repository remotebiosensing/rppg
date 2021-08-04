import cv2
import numpy as np
from tqdm import tqdm
from face_recognition import face_locations
from skimage.util import img_as_float


def Deepphys_preprocess_Video(path, flag):
    '''
    :param path: dataset path
    :param flag: face detect flag
    :return: [:,:,:0-2] : motion diff frame
             [:,:,:,3-5] : normalized frame
    '''
    cap = cv2.VideoCapture(path)
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_video = np.empty((frame_total - 1, 36, 36, 6))
    prev_frame = None
    j = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break
        if flag:
            rst, crop_frame = faceDetection(frame)
            if not rst:  # can't detect face
                return False, None
        else:
            crop_frame = frame[:, int(width / 2) - int(height / 2 + 1):int(height / 2) + int(width / 2), :]

        crop_frame = cv2.resize(crop_frame, dsize=(36, 36), interpolation=cv2.INTER_AREA)
        crop_frame = generate_Floatimage(crop_frame)

        if prev_frame is None:
            prev_frame = crop_frame
            continue
        raw_video[j, :, :, :3], raw_video[j, :, :, -3:] = preprocess_Image(prev_frame, crop_frame)
        prev_frame = crop_frame
        j += 1
    raw_video[:,:,:,3] = video_normalize(raw_video[:,:,:,3])
    cap.release()

    return True, raw_video


def PhysNet_preprocess_Video(path, flag):
    '''
    :param path: dataset path
    :param flag: face detect flag
    :return:
    '''
    cap = cv2.VideoCapture(path)
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_video = np.empty((frame_total, 128, 128, 3))
    j = 0
    with tqdm(total=frame_total, position=0, leave=True, desc=path) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if frame is None:
                break
            if flag:
                rst, crop_frame = faceDetection(frame)
                if not rst:  # can't detect face
                    return False, None
            else:
                crop_frame = frame[:, int(width / 2) - int(height / 2 + 1):int(height / 2) + int(width / 2), :]

            crop_frame = cv2.resize(crop_frame, dsize=(128, 128), interpolation=cv2.INTER_AREA)
            crop_frame = generate_Floatimage(crop_frame)

            raw_video[j] = crop_frame
            j += 1
            pbar.update(1)
        cap.release()

    split_raw_video = np.zeros(((frame_total // 32), 32, 128, 128, 3))
    index = 0
    for x in range(frame_total // 32):
        split_raw_video[x] = raw_video[index:index + 32]
        index = index + 32

    return True, split_raw_video


def faceDetection(frame):
    '''
    :param frame: one frame
    :return: cropped face image
    '''
    resized_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    face_location = face_locations(resized_frame)
    if len(face_location) == 0:  # can't detect face
        return False, None
    top, right, bottom, left = face_location[0]
    dst = resized_frame[top:bottom, left:right]
    return True, dst


def generate_Floatimage(frame):
    '''
    :param frame: roi frame
    :return: float value frame [0 ~ 1.0]
    '''
    dst = img_as_float(frame)
    # 왜 있지??
    dst = cv2.cvtColor(dst.astype('float32'), cv2.COLOR_BGR2RGB)
    dst[dst > 1] = 1
    dst[dst < 0] = 0
    return dst


def generate_MotionDifference(prev_frame, crop_frame):
    '''
    :param prev_frame: previous frame
    :param crop_frame: current frame
    :return: motion diff frame
    '''
    # motion input
    motion_input = (crop_frame - prev_frame) / (crop_frame + prev_frame)
    # TODO : need to diminish outliers [ clipping ]
    # motion_input = motion_input / np.std(motion_input)
    # TODO : do not divide each D frame, modify divide whole video's unit standard deviation
    return motion_input


def normalize_Image(frame):
    '''
    :param frame: image
    :return: normalized_frame
    '''
    return frame / np.std(frame)


def preprocess_Image(prev_frame, crop_frame):
    '''
    :param prev_frame: previous frame
    :param crop_frame: current frame
    :return: motion_differnceframe, normalized_frame
    '''
    return generate_MotionDifference(prev_frame, crop_frame), normalize_Image(prev_frame)


def ci99(motion_diff):
    max99 = np.mean(motion_diff) + (2.58 * (np.std(motion_diff) / np.sqrt(len(motion_diff))))
    min99 = np.mean(motion_diff) - (2.58 * (np.std(motion_diff) / np.sqrt(len(motion_diff))))
    motion_diff[motion_diff > max99] = max99
    motion_diff[motion_diff < min99] = min99
    return motion_diff


def video_normalize(channel):
    channel /= np.std(channel)
    return channel
