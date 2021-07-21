import cv2
import numpy as np
from face_recognition import face_locations
from skimage.util import img_as_float
import time


def preprocess_Video(path,flag):
    '''
    :param path: dataset path
    :param flag: face detect flag
    :return: [:,:,:0-2] : motion diff frame
             [:,:,:,3-5 : normalized frame
    '''
    cap = cv2.VideoCapture(path)
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
        if prev_frame is None:
            prev_frame = crop_frame
            continue
        raw_video[j, :, :, :3], raw_video[j, :, :, -3:] = preprocess_Image(prev_frame, crop_frame)
        prev_frame = crop_frame
        j += 1
    cap.release()
    return True, raw_video

def faceDetection(frame):
    '''
    :param frame: one frame
    :return: float face image [0.0~1.0]
    '''
    resized_frame = cv2.resize(frame,(0,0), fx=0.5,fy=0.5)
    face_location = face_locations(resized_frame)
    if len(face_location) == 0:  # can't detect face
        return False, None
    top,right,bottom,left = face_location[0]
    dst = img_as_float(frame[top:bottom,left:right])
    dst = cv2.resize(dst, dsize=(36, 36),interpolation=cv2.INTER_AREA)
    #왜 있지??
    dst = cv2.cvtColor(dst.astype('float32'), cv2.COLOR_BGR2RGB)
    dst[dst > 1] = 1
    dst[dst < 0] = 0
    return True, dst

def generate_MotionDifference(prev_frame, crop_frame):
    '''
    :param prev_frame: previous frame
    :param crop_frame: current frame
    :return: motion diff frame
    '''
    # motion input
    motion_input = (crop_frame - prev_frame) / (crop_frame + prev_frame)
    motion_input = motion_input / np.std(motion_input)
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
    return generate_MotionDifference(prev_frame,crop_frame), normalize_Image(crop_frame)
