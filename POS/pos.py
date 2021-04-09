#input : mean_RGB(X) - averaged R,G,B values of all pixels
# shape of X : [# of frames,3]

import matplotlib
# matplotlib.use("TkAgg")
from SkinDetector.skin_detector import skin_detector

from matplotlib import pyplot as plt

import os
import sys

sys.path.insert(0, './SkinDetector')
import pkg_resources

import numpy as np
import cv2
import dlib

import argparse

# EXTRACT PULSE
pulsedir = "/Volumes/MacMini-Backups/siw-db/live/pulse/"
start = 0
end = 450

framerate = 30

# FREQUENCY ANALYSIS
nsegments = 12

plot = False
image_show = True

left_increase_ratio = 0.05  # 5%
top_increase_ratio = 0.25  # 5%

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", default='/mnt/a7930c08-d429-42fa-a09e-15291e166a27/BVP_js/subject1/vid.avi',
                help="path to the (optional) video file")
args = vars(ap.parse_args())

if not args.get("video", False):
    from_webcam = True
    camera = cv2.VideoCapture(0)
    start = 0
    end = 450
# otherwise, load the video
else:
    camera = cv2.VideoCapture(args["video"])

video_file_path = args["video"]
video_file_name = os.path.basename(video_file_path)

start_index = start
end_index = end

# number of final frames
if end_index > 0:
    nb_frames = end_index - start_index

# loop on video frames
frame_counter = 0
i = start_index

detector = dlib.get_frontal_face_detector()

while (i >= start_index and i < end_index):
    (grabbed, frame) = camera.read()

    if not grabbed:
        continue

    print("Processing frame %d/%d...", i + 1, end_index)

    h, w, _ = frame.shape

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    if len(rects) == 0:
        continue

    if image_show:
        show_frame = frame.copy()

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
        print("Left, right, top, bottom: ", left, right, top, bottom)
        # print("Width and Height of bounding box : ",width,height)

        face_left = int(left - (left_increase_ratio / 2) * width)
        face_top = int(top - (top_increase_ratio) * height)
        # face_right = int(right + (area_increase_ratio/2)*width)
        # face_bottom = int(bottom + (area_increase_ratio/2)*height)

        face_right = right
        face_bottom = bottom

        print("Increased coordinates: ", face_left, face_right, face_top, face_bottom)

        if image_show:
            cv2.rectangle(show_frame, (left, top), (right, bottom), (255, 255, 0), 3)
            cv2.rectangle(show_frame, (face_left, face_top), (face_right, face_bottom), (0, 255, 0), 3)

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

        print("Mean RGB -> R = {0}, G = {1}, B = {2} ".format(r, g, b))

    if image_show:
        if h > w and h > 640:
            dim = (int(640 * (w / h)), 640)
            show_frame = cv2.resize(show_frame, dim, interpolation=cv2.INTER_LINEAR)
        if w > h and w > 640:
            dim = (640, int(640 * (h / w)))
            show_frame = cv2.resize(show_frame, dim, interpolation=cv2.INTER_LINEAR)

    # cv2.imshow("frame",show_frame)
    if (image_show):
        cv2.imshow("Masked face", masked_face)
        cv2.waitKey(1)
    frame_counter += 1
    i += 1
    # end loop

camera.release()
cv2.destroyAllWindows()

if plot:
    f = np.arange(0, mean_rgb.shape[0])
    plt.plot(f, mean_rgb[:, 0], 'r', f, mean_rgb[:, 1], 'g', f, mean_rgb[:, 2], 'b')
    plt.title("Mean RGB - Complete")
    plt.show()

framerate = 20

# Calculating window length(l)
#          - 1.6s : can capture at least one cardiac cycle in [40;240]bpm
l = int(framerate * 1.6) # 32

# Initialize
H = np.zeros(mean_rgb.shape[0])

for n in range(mean_rgb.shape[0]-l) :

    # Step1 :Spatial Averaging
    C = mean_rgb[n:n+l-1,:].T # Sliding each window

    # Step2 :Temporal normalization - Cn=diag(mean(C,2))^-1*C
    mean_C = np.mean(C, axis=1) # Mean
    diag_mean_C = np.diag(mean_C) # Diagonal
    diag_mean_C_inv = np.linalg.inv(diag_mean_C) # inverse
    Cn = np.matmul(diag_mean_C_inv, C)

    # Step3 :Projection(3D signal to 2D signal)
    projection_matrix = np.array([[0,1,-1],[-2,1,1]])
    S = np.matmul(projection_matrix, Cn)

    #Step4 :Tuning(2D signal to 1D signal)
    # - when the pulsatile variation dominates S(t), S1(t) and S2(t) appear in in-phase.
    std = np.array([1, np.std(S[0, :]) / np.std(S[1, :])])
    P = np.matmul(std, S)

    #Step5 :Overlap-adding
    H[n:n+l-1] = H[n:n+l-1] + (P-np.mean(P))/np.std(P)

if plot :
    from matplotlib import pyplot
    pyplot.plot(range(H.shape[0]), H, 'g')
    pyplot.title('Filtered green signal')
    pyplot.show()