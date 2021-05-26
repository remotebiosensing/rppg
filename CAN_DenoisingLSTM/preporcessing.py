import cv2
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt

Appearance_save = np.empty((1, 3, 36, 36))
Motion_save = np.empty((1, 3, 36, 36))
Target_save = np.empty(shape=(1,))

# subject_cnt = [1, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33, 34, 35, 36,
#                37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

# subject_cnt = [5]
# for sub_cnt in subject_cnt:
for sub_cnt in range(1,38):
    for folder in range(0,4):
        # Appearance_save = np.empty((1, 3, 36, 36))
        # Motion_save = np.empty((1, 3, 36, 36))
        # Target_save = np.empty(shape=(1,))
        # 동영상 파일에서 읽기
        # cap = cv2.VideoCapture("/home/js/Desktop/UBFC/subject" + str(sub_cnt) + "/vid.avi")
        cap = cv2.VideoCapture("/home/js/Desktop/COHFACE/" + str(sub_cnt) + "/" + str(folder) + "/data.avi")
        total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # get total frame size
        print(str(sub_cnt) + " : " + str(total_frame))
        prev_frame = None
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        Appearance_list = []
        Motion_list = []

        while cap.isOpened():
            ret, frame = cap.read()
            if frame is None:
                break
            shape = frame.shape

            # dst = cv2.resize(frame[:, int(width / 2) - int(height / 2 + 1):int(height / 2) + int(width / 2), :],
            #                  dsize=(36, 36), interpolation=cv2.INTER_CUBIC)
            dst = cv2.resize(frame[50:350,150:490],
                             dsize=(36, 36), interpolation=cv2.INTER_CUBIC)
            # cv2.imshow('dst', dst)

            if prev_frame is None:
                print(prev_frame)
                prev_frame = dst
                continue

            img1 = torch.from_numpy(np.array(prev_frame).astype(np.float32))
            img2 = torch.from_numpy(np.array(dst).astype(np.float32))
            img1 = img1.permute(2, 0, 1)
            img2 = img2.permute(2, 0, 1)

            M = torch.div(img2 - img1, img1 + img2 + 1)
            Motion_list.append(M.tolist())
            M_test = cv2.resize(M.permute(1,2,0).numpy(), dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
            cv2.imshow('Appearance',M.permute(1,2,0).numpy())
            # plt.imshow(M.permute(1,2,0).numpy())
            # plt.show(block=False)
            # plt.pause(1)
            # plt.close()

            A = img1 / 255
            A = torch.sub(A, torch.mean(A, (1, 2)).view(3, 1, 1))
            Appearance_list.append(A.tolist())
            # cv2.imshow('Appearance',A.permute(1,2,0).numpy())
            # print(A)
            # plt.imshow(A.permute(1,2,0))
            # plt.show(block=False)
            # plt.pause(1)
            # plt.close()
            prev_frame = dst

            key = cv2.waitKey(1) & 0xFF
            if (key == 27):
                break
        cap.release()

        # f = open("/home/js/Desktop/UBFC/subject" + str(sub_cnt) + "/ground_truth.txt", 'r')
        # f_read = f.read().split('\n')
        # label = ' '.join(f_read[0].split()).split()
        # label = list(map(float, label))
        # delta_label = []
        # for i in range(len(label) - 1):
        #     delta_label.append(label[i + 1] - label[i])
        # delta_label = np.array(delta_label).astype('float32')
        # delta_std = delta_label.copy()
        # f.close()
        # load hdf5 file

        f = h5py.File("/home/js/Desktop/COHFACE/" + str(sub_cnt) + "/" + str(folder) + "/data.hdf5", 'r')
        pulse_group_key = list(f.keys())[0]
        rr_group_key = list(f.keys())[1]
        time_group_key = list(f.keys())[2]

        pulse = list(f[pulse_group_key])
        rr = list(f[rr_group_key])
        time = list(f[time_group_key])
        f.close()
        pulse = np.interp(np.arange(0, float(total_frame)),
                          np.linspace(0, float(total_frame), num=len(pulse)),
                          pulse)
        delta_pulse = []
        for i in range(len(pulse) - 1):
            delta_pulse.append(pulse[i + 1] - pulse[i])
        delta_pulse = np.array(delta_pulse).astype('float32')

        part = 0
        window = 32
        while part < (len(delta_pulse) // window) - 1:
            delta_pulse[part * window:(part + 1) * window] /= np.std(delta_pulse[part * window:(part + 1) * window])
            part += 1
        if len(delta_pulse) % window != 0:
            delta_pulse[part * window:] /= np.std(delta_pulse[part * window:])
        print(delta_pulse)

        Appearance_list = np.array(Appearance_list)
        Motion_list = np.array(Motion_list)

        Appearance_save = np.concatenate((Appearance_save, Appearance_list), axis=0)
        Motion_save = np.concatenate((Motion_save, Motion_list), axis=0)
        Target_save = np.concatenate((Target_save, delta_pulse), axis=0)

        print("-----------------------------------------------------------------------------------------------------")
        print(str(sub_cnt) + " finish - " + str(folder))
        print("-----------------------------------------------------------------------------------------------------")

        Appearance_save = np.delete(Appearance_save, 0, 0)
        Motion_save = np.delete(Motion_save, 0, 0)
        Target_save = np.delete(Target_save, 0)
        print('test')

# np.savez_compressed("./preprocessing/dataset/" + "COHFACE_test_" + str(sub_cnt) + "_" + str(folder),
#                     A=Appearance_save, M=Motion_save, T=Target_save, N=Name_save)
np.savez_compressed("./preprocessing/dataset/" + "CCOHFACE_trainset_face",
                    A=Appearance_save, M=Motion_save, T=Target_save)
