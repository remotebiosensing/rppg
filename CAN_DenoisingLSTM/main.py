import cv2
import numpy as np
import torch
import torchvision
from skimage.util import img_as_float
import time

start_total = time.time()

Appearance_save = np.empty((1, 3, 36, 36))
Motion_save = np.empty((1, 3, 36, 36))
Target_save = np.empty(shape=(1,))

subject_cnt = [1, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33, 34, 35, 36,
               37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
# subject_cnt = [1, 3]

for sub_cnt in subject_cnt:
    # 동영상 파일에서 읽기
    cap = cv2.VideoCapture("/mnt/a7930c08-d429-42fa-a09e-15291e166a27/BVP_js/subject" + str(sub_cnt) + "/vid.avi")
    cnt = 0
    Previous_frame = None
    Appearnce_list = []
    Motion_list = []

    while cap.isOpened():
        # 카메라 프레임 읽기
        ret, frame = cap.read()
        start_frame = time.time()
        if frame is None:
            break
        shape = frame.shape
        src_cols = shape[0]
        src_rows = shape[1]
        dst = frame[:, int(src_rows / 2) - int(src_cols / 2 + 1):int(src_cols / 2) + int(src_rows / 2), :]
        dst = cv2.resize(dst, dsize=(36, 36), interpolation=cv2.INTER_CUBIC)
        if Previous_frame is None:
            Previous_frame = np.zeros_like(dst)
        M = torch.div(torchvision.transforms.ToTensor()(dst) - torchvision.transforms.ToTensor()(Previous_frame),
                      torchvision.transforms.ToTensor()(dst) + torchvision.transforms.ToTensor()(Previous_frame) + 1)

        Motion_list.append(M.tolist())

        Appearnce = torchvision.transforms.ToTensor()(dst)
        # Appearnce = torch.tensor(dst/255)
        Appearnce = torch.sub(Appearnce, torch.mean(Appearnce, (1, 2)).view(3, 1, 1))
        Appearnce_list.append(Appearnce.tolist())

        cnt = cnt + 1

        Previous_frame = dst

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()

    print("-----------------------------------------------------------------------------------------------------")
    print(str(sub_cnt) + " finish")
    print("frame calculate time : " + str(time.time() - start_frame))
    print("-----------------------------------------------------------------------------------------------------")
    f = open("/mnt/a7930c08-d429-42fa-a09e-15291e166a27/BVP_js/subject" + str(sub_cnt) + "/ground_truth.txt", 'r')
    f_read = f.read().split('\n')
    label = ' '.join(f_read[0].split()).split()
    label = list(map(float, label))
    delta_label = []
    for i in range(len(label) - 1):
        delta_label.append(label[i + 1] - label[i])
    delta_label = np.array(delta_label).astype('float32')
    f.close()

    Appearnce_list = np.array(Appearnce_list)
    Motion_list = np.array(Motion_list)

    Appearance_save = np.concatenate((Appearance_save, Appearnce_list), axis=0)
    Motion_save = np.concatenate((Motion_save, Motion_list), axis=0)

    Target_save = np.concatenate((Target_save, delta_label), axis=0)

Appearance_save = np.delete(Appearance_save, 0, 0)
Motion_save = np.delete(Motion_save, 0, 0)
Target_save = np.delete(Target_save, 0)

print("total_save time : " + str(time.time() - start_total))
np.savez_compressed("/mnt/a7930c08-d429-42fa-a09e-15291e166a27/BVP_js/DATASET_2/M/subject_total/" + "subject_test_4",
                    A=Appearance_save, M=Motion_save, T=Target_save)

