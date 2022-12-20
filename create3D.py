import matplotlib.pyplot as plt
import os
import json
import cv2
import numpy as np
import mediapipe as mp
import skimage
from skimage.transform import PiecewiseAffineTransform, warp

def imshow(img):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img)
    return ax


uv_path = "./uv_datas/uv_map.json" #taken from https://github.com/spite/FaceMeshFaceGeometry/blob/353ee557bec1c8b55a5e46daf785b57df819812c/js/geometry.js
uv_map_dict = json.load(open(uv_path))
uv_map = np.array([ (uv_map_dict["u"][str(i)],uv_map_dict["v"][str(i)]) for i in range(468)])


H_new,W_new = 512,512
# read video and get the frame
cap = cv2.VideoCapture("/media/hdd1/UBFC/subject1/vid.avi")
out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (W_new,H_new))
# read frame until video is over

cnt = 0
total = 200

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    H, W, C = frame.shape

    with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(frame)

    face_landmarks = results.multi_face_landmarks[0]
    keypoints = np.array(
        [(W * point.x, H * point.y) for point in face_landmarks.landmark[0:468]])  # after 468 is iris or something else
    # ax = imshow(frame)
    # ax.plot(keypoints[:, 0], keypoints[:, 1], '.b', markersize=2)
    # plt.show()


    keypoints_uv = np.array([(W_new*x, H_new*y) for x,y in uv_map])

    tform = PiecewiseAffineTransform()
    tform.estimate(keypoints_uv,keypoints)
    texture = warp(frame, tform, output_shape=(H_new,W_new))
    texture = (255*texture).astype(np.uint8)

    # ax = imshow(texture)
    # ax.plot(keypoints_uv[:, 0], keypoints_uv[:, 1], '.b', markersize=2)
    # plt.show()
    out.write(texture)

    cnt+=1
    print(cnt)
    if cnt == total:
        break

# make video using opencv and save it

# for i in range(100):

out.release()