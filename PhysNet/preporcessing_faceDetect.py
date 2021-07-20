import cv2
import numpy as np
import Facedetect
from skimage.util import img_as_float
import h5py


def face_detect(frame):
    face_detector = Facedetect.Facedetect()
    src_cols = frame.shape[0]
    src_rows = frame.shape[1]
    ratio_cols = src_cols / 800
    ratio_rows = src_rows / 1200
    out_result = face_detector.detect(cv2.resize(frame, dsize=(128, 128), interpolation=cv2.INTER_CUBIC))
    parameter_width = 30
    parameter_height = 100
    for i in range(len(out_result)):
        if len(out_result[i]) < 15:
            break
        for j in range(len(out_result[i])):
            if j % 2 is 1:
                out_result[i][j] = int(out_result[i][j] * ratio_cols)
            else:
                out_result[i][j] = int(out_result[i][j] * ratio_rows)
    dst = cv2.resize(
        img_as_float(frame[out_result[0][1] - parameter_height:out_result[0][3] + int(parameter_height / 2),
                     out_result[0][12] - parameter_width:out_result[0][14] + parameter_width]), dsize=(128, 128),
        interpolation=cv2.INTER_AREA)
    dst = cv2.resize(
        img_as_float(frame[out_result[0][1]:out_result[0][3],
                     out_result[0][12]:out_result[0][14]]), dsize=(128, 128),
        interpolation=cv2.INTER_AREA)
    # cv2.imshow("test", dst)
    dst = cv2.cvtColor(dst.astype('float32'), cv2.COLOR_BGR2RGB)
    dst[dst > 1] = 1
    dst[dst < (1 / 255)] = 1 / 255
    return dst


def normalize_difference(prev_frame, crop_frame):
    # motion input
    M = (crop_frame - prev_frame) / (crop_frame + prev_frame)
    M = M / np.std(M)
    # appearance input
    A = crop_frame / np.std(crop_frame)

    return M, A


class DatasetPhysNetUBFC:
    def __init__(self):
        self.path = "/media/hdd1/UBFC"
        # self.subject_cnt = [1, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33,
        #                     34, 35, 36, 37, 38, 39, 40, 42, 43, 44]
        self.subject_cnt = [48]

    def __call__(self):
        output_video = np.zeros((1, 32, 128, 128, 3))
        output_label = np.zeros((1, 32))
        for sub_cnt in self.subject_cnt:
            raw_video, total_frame = preprocess_raw_video(self.path + "/subject" + str(sub_cnt) + "/vid.avi")
            output_video = np.concatenate((output_video, raw_video), axis=0)
            label = self.preprocess_label(self.path + "/subject" + str(sub_cnt) + "/ground_truth.txt")
            output_label = np.concatenate((output_label, label), axis=0)
            print(sub_cnt)
        output_video = np.delete(output_video, 0, 0)
        output_label = np.delete(output_label, 0, 0)
        # Save Data
        data_file = h5py.File(
            '/media/hdd1/js_dataset/UBFC_PhysNet/UBFC_test_Data_48.hdf5', 'w')
        data_file.create_dataset('output_video', data=output_video)
        data_file.create_dataset('output_label', data=output_label)
        data_file.close()
        return output_video, output_label

    def preprocess_label(self, path):
        # Load input
        f = open(path, 'r')
        f_read = f.read().split('\n')
        label = ' '.join(f_read[0].split()).split()
        label = list(map(float, label))
        label = np.array(label).astype('float32')
        split_raw_label = np.zeros(((len(label) // 32), 32))
        index = 0
        for i in range(len(label) // 32):
            split_raw_label[i] = label[index:index + 32]
            index = index + 32
        return split_raw_label


def preprocess_raw_video(path):
    cap = cv2.VideoCapture(path)
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    raw_video = np.empty((frame_total, 128, 128, 3))
    j = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break
        crop_frame = face_detect(frame)
        raw_video[j] = crop_frame
        j += 1
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    cap.release()
    split_raw_video = np.zeros(((frame_total // 32), 32, 128, 128, 3))
    index = 0
    for x in range(frame_total // 32):
        split_raw_video[x] = raw_video[index:index + 32]
        index = index + 32
    return split_raw_video, frame_total


dataset = DatasetPhysNetUBFC()
dataset()
