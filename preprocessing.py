import numpy as np
import cv2
from skimage.util import img_as_float

def preprocess_raw_video(videoFilePath, dim=36):
    #########################################################################
    # set up
    t = []
    i = 0
    vidObj = cv2.VideoCapture(videoFilePath);
    totalFrames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT)) # get total frame size
    Xsub = np.zeros((totalFrames, dim, dim, 3), dtype = np.float32)
    height = vidObj.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = vidObj.get(cv2.CAP_PROP_FRAME_WIDTH)
    success, img = vidObj.read()
    dims = img.shape
    #########################################################################
    # Crop each frame size into dim x dim
    while success:
        t.append(vidObj.get(cv2.CAP_PROP_POS_MSEC))# current timestamp in milisecond
        vidLxL = cv2.resize(img_as_float(img[:, int(width/2)-int(height/2 + 1):int(height/2)+int(width/2), :]), (dim, dim), interpolation = cv2.INTER_AREA)
        vidLxL = cv2.rotate(vidLxL, cv2.ROTATE_90_COUNTERCLOCKWISE) # rotate 90 degree
        vidLxL = cv2.cvtColor(vidLxL.astype('float32'), cv2.COLOR_BGR2RGB)
        vidLxL[vidLxL > 1] = 1
        vidLxL[vidLxL < (1/255)] = 1/255
        Xsub[i, :, :, :] = vidLxL
        success, img = vidObj.read() # read the next one
        i = i + 1
    #########################################################################
    # Normalized Frames in the motion branch
    normalized_len = len(t) - 1
    dXsub = np.zeros((normalized_len, dim, dim, 3), dtype = np.float32)
    for j in range(normalized_len - 1):
        dXsub[j, :, :, :] = (Xsub[j+1, :, :, :] - Xsub[j, :, :, :]) / (Xsub[j+1, :, :, :] + Xsub[j, :, :, :])
    dXsub = dXsub / np.std(dXsub)
    #########################################################################
    # Normalize raw frames in the apperance branch
    Xsub = Xsub - np.mean(Xsub)
    #Xsub = Xsub  / np.std(Xsub)
    Xsub = Xsub[:totalFrames-1, :, :, :]
    #########################################################################
    # Plot an example of data after preprocess
    dXsub = np.concatenate((dXsub, Xsub), axis = 3);
    return dXsub

def preprocess_label(label_path):
    f = open(label_path, 'r')
    f_read = f.read().split('\n')
    label = ' '.join(f_read[0].split()).split()
    label = list(map(float,label))
    delta_label = []
    for i in range(len(label)-1):
        delta_label.append(label[i+1]-label[i])
    delta_label = np.array(delta_label).astype('float32')

    # label = np.delete(label, len(label)-1,axis=0)
   # rr = rr.reshape(cnt, 1, 1, 1)
    f.close()
    return delta_label

#subject_cnt = [1, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33, 34, 35, 36,
#               37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
subject_cnt = [1]

def generate_npz(root_dir="/mnt/a7930c08-d429-42fa-a09e-15291e166a27/BVP_js/subject"):

    target_image = np.empty(shape=(1,36,36,6))
    target_label = np.empty(shape =(1,))

    for sub_cnt in subject_cnt:
        print(str(sub_cnt)+"=======")
        dXsub = preprocess_raw_video(root_dir+str(sub_cnt)+"/vid.avi")
        label = preprocess_label(root_dir+str(sub_cnt)+"/ground_truth.txt")
        target_image = np.concatenate((target_image,dXsub),axis=0)
        target_label = np.concatenate((target_label,label),axis=0)

    target_image = np.delete(target_image,0,0)
    target_label = np.delete(target_label,0)
    np.savez_compressed("./subject_1",A=target_image[:,:,:,-3:],M=target_image[:,:,:,:3],T=target_label)



generate_npz()