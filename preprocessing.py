import numpy as np
import cv2
from skimage.util import img_as_float
from bvpdataset import bvpdataset
import torch
from torch.utils.data import Dataset


class DatasetDeepPhysUBFC():
    def __init__(self, video_path, img_size, preprocessing, train):
        # self.subject_cnt = [1, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33, 34, 35, 36,
        #               37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
        self.subject_cnt = [1, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33, 34, 35, 36,
                      37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
        self.root_dir = video_path
        self.dim = img_size
        self.preprocessing = preprocessing
        self.train = train

    def __call__(self):
        if self.preprocessing is False:
            if self.train is True:
                dataset = np.load("./subject_test.npz")
                print("Complete Dataset : subject_train.npz")
                dataset = bvpdataset(A=dataset['A'], M=dataset['M'], T=dataset['T'])
                tot = int(len(dataset)) // 128
                bias = tot // 10
                train_set, val_set = torch.utils.data.random_split(dataset,
                                                                   [128 * bias * 8, int(len(dataset)) - 128 * bias * 8],
                                                                   # [int(len(dataset) * 0.8),
                                                                   #  int(len(dataset) * 0.2 + 1)],
                                                                   generator=torch.Generator().manual_seed(1))
                return train_set, val_set
            else:
                dataset = np.load("./subject_test.npz")
                print("Complete Dataset : subject_test.npz")
                test_set = bvpdataset(A=dataset['A'], M=dataset['M'], T=dataset['T'])
                return test_set
        else:
            target_image = np.empty(shape=(1, 36, 36, 6))
            target_label = np.empty(shape=(1,))

            for sub_cnt in self.subject_cnt:

                print("Preprocessing : "+ str(sub_cnt) + "=======")
                dXsub = self.preprocess_raw_video(self.root_dir + str(sub_cnt) + "/vid.avi")
                label = self.preprocess_label(self.root_dir + str(sub_cnt) + "/ground_truth.txt")
                target_image = np.concatenate((target_image, dXsub), axis=0)
                target_label = np.concatenate((target_label, label), axis=0)

                target_image = np.delete(target_image, 0, 0)
                target_label = np.delete(target_label, 0)
                np.savez_compressed("./subject_train", A=target_image[:, :, :, -3:], M=target_image[:, :, :, :3], T=target_label)
                dataset = bvpdataset(A=target_image[:, :, :, -3:], M=target_image[:, :, :, :3], T=target_label)
                tot = int(len(dataset)) // 128
                bias = tot // 10
                train_set, val_set = torch.utils.data.random_split(dataset,
                                                                   [128 * bias * 8, int(len(dataset)) - 128 * bias * 8],
                                                                   # [int(len(dataset) * 0.8),
                                                                   #  int(len(dataset) * 0.2 + 1)],
                                                                   generator=torch.Generator().manual_seed(1))
            return train_set, val_set

    def preprocess_raw_video(self, image_root):
        #########################################################################
        # set up
        t = []
        i = 0
        vidObj = cv2.VideoCapture(image_root);
        totalFrames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))  # get total frame size
        Xsub = np.zeros((totalFrames, self.dim, self.dim, 3), dtype=np.float32)
        height = vidObj.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = vidObj.get(cv2.CAP_PROP_FRAME_WIDTH)
        success, img = vidObj.read()
        #########################################################################
        # Crop each frame size into dim x dim
        while success:
            t.append(vidObj.get(cv2.CAP_PROP_POS_MSEC))  # current timestamp in milisecond
            vidLxL = cv2.resize(
                img_as_float(img[:, int(width / 2) - int(height / 2 + 1):int(height / 2) + int(width / 2), :]),
                (self.dim, self.dim), interpolation=cv2.INTER_AREA)
            vidLxL = cv2.rotate(vidLxL, cv2.ROTATE_90_COUNTERCLOCKWISE)  # rotate 90 degree
            vidLxL = cv2.cvtColor(vidLxL.astype('float32'), cv2.COLOR_BGR2RGB)
            vidLxL[vidLxL > 1] = 1
            vidLxL[vidLxL < (1 / 255)] = 1 / 255
            Xsub[i, :, :, :] = vidLxL
            success, img = vidObj.read()  # read the next one
            i = i + 1
        #########################################################################
        # Normalized Frames in the motion branch
        normalized_len = len(t) - 1
        dXsub = np.zeros((normalized_len, self.dim, self.dim, 3), dtype=np.float32)
        for j in range(normalized_len - 1):
            dXsub[j, :, :, :] = (Xsub[j + 1, :, :, :] - Xsub[j, :, :, :]) / (Xsub[j + 1, :, :, :] + Xsub[j, :, :, :])
        dXsub = dXsub / np.std(dXsub)
        #########################################################################
        # Normalize raw frames in the apperance branch
        Xsub = Xsub - np.mean(Xsub)
        # Xsub = Xsub  / np.std(Xsub)
        Xsub = Xsub[:totalFrames - 1, :, :, :]
        #########################################################################
        # Plot an example of data after preprocess
        dXsub = np.concatenate((dXsub, Xsub), axis=3);

        return dXsub

    def preprocess_label(self, label_path):
        f = open(label_path, 'r')
        f_read = f.read().split('\n')
        label = ' '.join(f_read[0].split()).split()
        label = list(map(float, label))
        delta_label = []
        for i in range(len(label) - 1):
            delta_label.append(label[i + 1] - label[i])
        delta_label = np.array(delta_label).astype('float32')

        f.close()
        return delta_label
