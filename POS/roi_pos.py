import os
import cv2
import dlib
import numpy as np

import argparse
import face_recognition

from matplotlib import pyplot

def main(video_path):
    # EXTRACT PULSE
    start = 0
    end = 450

    framerate = 30

    # FREQUENCY ANALYSIS
    nsegments = 12

    plot =  False
    image_show = False
    print_log = False

    left_increase_ratio = 0.05  # 5%
    top_increase_ratio = 0.25  # 5%

    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", default = video_path, help = "path to the (optional) video file")
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


    while (i >= start_index and i < end_index):
        (grabbed, frame) = camera.read()

        if not grabbed:
            continue

        h,w,_ = frame.shape

        face_landmarks_list=face_recognition.face_landmarks(frame)

        dict_face_to_list = sum(list(face_landmarks_list[0].values()), [])

        landmark_list=[]

        for k in range(len(dict_face_to_list)) :
            landmark =list(dict_face_to_list[k])
            landmark_list.append(landmark)

        # Define each cheek
        right_cheek_arr = frame[landmark_list[28][1]:landmark_list[33][1], landmark_list[35][0]: landmark_list[12][0]]
        left_cheek_arr = frame[landmark_list[28][1]: landmark_list[33][1], landmark_list[5][0]: landmark_list[31][0]]  # left cheeks

        # Show cheek boxes
        if image_show:
            right_cheek_box = cv2.rectangle(frame, (landmark_list[35][0], landmark_list[28][1]), (landmark_list[12][0],landmark_list[33][1]),(255,0,0),2)
            left_cheek_box = cv2.rectangle(frame, (landmark_list[31][0], landmark_list[28][1]), (landmark_list[4][0], landmark_list[33][1]), (0, 255, 0), 2)
            cv2.imshow('test', left_cheek_box)
            cv2.imshow('test', right_cheek_box)

        # Show face landmarks (indexes(72) & location points)

        #frame_resize = cv2.resize(face, (4*frame.shape[1], 4*frame.shape[0]), interpolation=cv2.INTER_LINEAR)

        for index in range(len(landmark_list)) :
            cv2.line(frame, (landmark_list[index][0]*4, landmark_list[index][1]*4) ,(landmark_list[index][0]*4, landmark_list[index][1]*4),(255,0,0),8)
            cv2.putText(frame, str(index), ((landmark_list[index][0])*4, (landmark_list[index][1])*4), cv2.FONT_HERSHEY_PLAIN,1,(0,0,0), 2)

        if image_show :
            cv2.imshow('Indexes and Points',frame)
            cv2.waitKey(1)

        number_of_right_cheek_pixels = right_cheek_arr.shape[0]*right_cheek_arr.shape[1]
        number_of_left_cheek_pixels = left_cheek_arr.shape[0] * left_cheek_arr.shape[1]

        number_of_total_cheek_pixels = number_of_right_cheek_pixels + number_of_left_cheek_pixels

        r = (np.sum(right_cheek_arr[:, :, 2]) + np.sum(left_cheek_arr[:, :, 2])) /number_of_total_cheek_pixels
        g = (np.sum(right_cheek_arr[:, :, 1]) + np.sum(left_cheek_arr[:, :, 1])) /number_of_total_cheek_pixels
        b = (np.sum(right_cheek_arr[:, :, 0]) + np.sum(left_cheek_arr[:, :, 0])) / number_of_total_cheek_pixels

        if frame_counter==0:
            mean_rgb = np.array([r,g,b])
        else:
            mean_rgb = np.vstack((mean_rgb,np.array([r,g,b])))

        if print_log:
            print("Mean RGB -> R = {0}, G = {1}, B = {2} ".format(r,g,b))

        if image_show:
            if h>w and h>640:
                    dim = (int(640 * (w/h)),640)
                    show_frame = cv2.resize(show_frame, dim, interpolation = cv2.INTER_LINEAR)
            if w>h and w>640:
                    dim = (640, int(640 * (h/w)))
                    show_frame = cv2.resize(show_frame, dim, interpolation = cv2.INTER_LINEAR)

        sub_name = video_path.split('/')
        sub = sub_name[-2]

        if(image_show):
            cv2.imshow( sub, frame)
            cv2.waitKey(1)
        frame_counter +=1
        i += 1
        #end loop

    camera.release()
    cv2.destroyAllWindows()

    # Calculating window length(l)
    #          - 1.6s : can capture at least one cardiac cycle in [40;240]bpm
    l = int(framerate * 1.6) # 48

    # Initialize H with 0(len =450)
    H = np.zeros(mean_rgb.shape[0])

    for n in range(mean_rgb.shape[0]-l) :

        # Step1 :Spatial Averaging
        C = mean_rgb[n:n+l-1,:].T # Sliding each window

        # Step2 :Temporal normalization - Cn=diag(mean(C,2))^-1*C
        mean_C = np.mean(C, axis=1) # Mean
        diag_mean_C = np.diag(mean_C) # Diagonal
        diag_mean_C_inv = np.linalg.inv(diag_mean_C) # Inverse
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

    if print_log:
        print("Pulse",H)
    signal = H
    if print_log :
        print("Pulse shape", H.shape)

    # FFT to find the maximum frequency
    # find the segment length, such that we have 8 50% overlapping segments (Matlab's default)
    segment_length = (2*signal.shape[0]) // (nsegments + 1)

    if print_log :
        print("nperseg",segment_length)

    if plot :
        pyplot.plot(range(signal.shape[0]), signal, 'g')
        pyplot.title('Filtered green signal')
        pyplot.show()

    # Transfer to frequency domain to find ROI
    from scipy.signal import welch
    signal = signal.flatten()
    green_f, green_psd = welch(signal, framerate, 'flattop', nperseg=300)
    # print("Green F, Shape",green_f,green_f.shape)
    # print("Green PSD, Shape",green_psd,green_psd.shape)

    green_psd = green_psd.flatten()
    #Set the range of target freqency range
    first = np.where(green_f > 0.9)[0] #0.8 for 300 frames
    last = np.where(green_f < 2.2)[0]
    first_index = first[0]
    last_index = last[-1]
    range_of_interest = range(first_index, last_index + 1, 1)

    # print("Range of interest",range_of_interest)
    max_idx = np.argmax(green_psd[range_of_interest])
    f_max = green_f[range_of_interest[max_idx]]

    hr = f_max*60.0
    print("Heart rate = {0}".format(hr))

    if plot:
        pyplot.semilogy(green_f, green_psd, 'g')
        xmax, xmin, ymax, ymin = pyplot.axis()
        pyplot.vlines(green_f[range_of_interest[max_idx]], ymin, ymax, color='red')
        pyplot.title('Power spectrum of the green signal (HR = {0:.1f})'.format(hr))
        pyplot.show()

    return hr

if __name__ == "__main__":

    # Select data_except 15, 16, 25, 43,49
    file_list =  [1,3,4,8,9,10,11,12,13,14,17,20,22,23,24,26,27,30,32,31,33,34,35,36,37,38,39,40,41,42,44,45,46,47,48]
    files = list(map(str, file_list))
    value = []

    for i in files:
        path = '../../js/Desktop/UBFC/subject'+i+'/vid.avi'
        print("PROCESSING : ",i,"/")
        hr=main(path)

        # Save result as a txt file
        f= open('./Result_HR.txt', 'a')
        f.write(str(hr))
        f.write("\n")

    f.close()
