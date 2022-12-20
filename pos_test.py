import matplotlib
#matplotlib.use("TkAgg")
# matplotlib.use("MacOSX")
from matplotlib import pyplot as plt

import os
import sys
sys.path.insert(0, './unused/SkinDetector')
import pkg_resources

import numpy as np
import cv2
import dlib

from imutils.video import VideoStream
from imutils import face_utils
import imutils

import argparse
import unused.SkinDetector.skin_detector as skin_detector

video_file_path = './output.avi'

left_increase_ratio = 0.05
top_increase_ratio = 0.25

camera = cv2.VideoCapture(video_file_path)

start_idx = 0
end_idx = 200

framerate = 30

if end_idx > 0:
    nb_frames = end_idx - start_idx

frame_counter = 0
i = start_idx

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./unused/SkinDetector/shape_predictor_68_face_landmarks.dat')


while (i >= start_idx and i < end_idx):

    (grabbed, frame) = camera.read()

    #image vertical flip using cv2
    new_frame = cv2.flip(frame, 1)

    frame[:,0:256, :] = new_frame[:, 0:256, :]

    if not grabbed:
        continue

    # print('Processing frame: {}'.format(i))

    h, w, c = frame.shape



    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    if len(rects) == 0:
        print('No face detected')
        continue

    if (len(rects) > 0):
        rect = rects[0]
        '''          
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        for counter,(x, y) in enumerate(shape):
            cv2.circle(show_frame, (x, y), 4, (0, 0, 255), -1)
            cv2.putText(show_frame,str(counter),(x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(255,255,255),1)
        '''

        left, right, top, bottom = rect.left(), rect.right(), rect.top(), rect.bottom()
        width = abs(right - left)
        height = abs(bottom - top)
        # print("Left, right, top, bottom: ", left, right, top, bottom)
        # print("Width and Height of bounding box : ",width,height)

        face_left = int(left - (left_increase_ratio / 2) * width)
        face_top = int(top - (top_increase_ratio) * height)
        # face_right = int(right + (area_increase_ratio/2)*width)
        # face_bottom = int(bottom + (area_increase_ratio/2)*height)

        face_right = right
        face_bottom = bottom

        # print("Increased coordinates: ", face_left, face_right, face_top, face_bottom)


        if face_left < 0:
            face_left = 0
        if face_top < 0:
            face_top = 0
        if face_right > w:
            face_right = w
        if face_bottom > h:
            face_bottom = h

        face = frame[face_top:face_bottom, face_left:face_right]

        if (face.size == 0):
            continue
        #    continue
        # Extract face skin pixels
        mask = skin_detector.process(face)

        # print("Mask shape: ",mask.shape)
        masked_face = cv2.bitwise_and(face, face, mask=mask)
        number_of_skin_pixels = np.sum(mask > 0)

        # compute mean
        r = np.sum(masked_face[:, :, 2]) / number_of_skin_pixels
        g = np.sum(masked_face[:, :, 1]) / number_of_skin_pixels
        b = np.sum(masked_face[:, :, 0]) / number_of_skin_pixels

        if frame_counter == 0:
            mean_rgb = np.array([r, g, b])
        else:
            mean_rgb = np.vstack((mean_rgb, np.array([r, g, b])))

        # print("Mean RGB -> R = {0}, G = {1}, B = {2} ".format(r, g, b))
    frame_counter += 1
    i += 1


    l = int(framerate * 1.6)

    H = np.zeros(mean_rgb.shape[0])

    for t in range(0, (mean_rgb.shape[0] - l)):
        # t = 0
        # Step 1: Spatial averaging
        C = mean_rgb[t:t + l - 1, :].T
        # C = mean_rgb.T
        # print("C shape", C.shape)
        # print("t={0},t+l={1}".format(t, t + l))
        # Step 2 : Temporal normalization
        mean_color = np.mean(C, axis=1)
        # print("Mean color", mean_color)

        diag_mean_color = np.diag(mean_color)
        # print("Diagonal",diag_mean_color)

        diag_mean_color_inv = np.linalg.inv(diag_mean_color)
        # print("Inverse",diag_mean_color_inv)

        Cn = np.matmul(diag_mean_color_inv, C)
        # Cn = diag_mean_color_inv@C
        # print("Temporal normalization", Cn)
        # print("Cn shape", Cn.shape)
        projection_matrix = np.array([[0, 1, -1], [-2, 1, 1]])
        S = np.matmul(projection_matrix, Cn)
        # S = projection_matrix@Cn
        # print("S matrix", S)
        # print("S shape", S.shape)
        if False:
            f = np.arange(0, S.shape[1])
            # plt.ylim(0,100000)
            plt.plot(f, S[0, :], 'c', f, S[1, :], 'm')
            plt.title("Projection matrix")
            plt.show()

        # Step 4:
        # 2D signal to 1D signal
        std = np.array([1, np.std(S[0, :]) / np.std(S[1, :])])
        # print("std", std)
        P = np.matmul(std, S)
        # P = std@S
        # print("P", P)
        if False:
            f = np.arange(0, len(P))
            plt.plot(f, P, 'k')
            plt.title("Alpha tuning")
            plt.show()

        # Step 5: Overlap-Adding
        H[t:t + l - 1] = H[t:t + l - 1] + (P - np.mean(P)) / np.std(P)

    # print("Pulse", H)
    signal = H

    if i == 200:
        f = np.arange(0,200)
        plt.plot(f,H)
        plt.show()
        break

    # print("Pulse shape", H.shape)

    segment_length = (2 * signal.shape[0]) // (12 + 1)


