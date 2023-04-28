import matplotlib.pyplot as plt
import json
import cv2
import numpy as np
from rppg.preprocessing.image_preprocess import faceUnwrapping
def imshow(img):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img)
    return ax


uv_path = "uv_datas/uv_map.json"  #taken from https://github.com/spite/FaceMeshFaceGeometry/blob/353ee557bec1c8b55a5e46daf785b57df819812c/js/geometry.js
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
    texture = faceUnwrapping(frame, (W_new, H_new), uv_map, 0)
    tmp1 = faceUnwrapping(frame, (W_new, H_new), uv_map, 1)
    tmp2 = faceUnwrapping(frame, (W_new, H_new), uv_map, 2)
    tmp3 = faceUnwrapping(frame, (W_new, H_new), uv_map, 3)

    out.write(texture)

    cnt+=1
    print(cnt)
    if cnt == total:
        break

# make video using opencv and save it

# for i in range(100):

out.release()