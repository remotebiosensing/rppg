#input : mean_RGB(X) - averaged R,G,B values of all pixels
# shape of X : [# of frames,3]          


import numpy as np
import cv2

framerate = 20

# Calculating window length(l)
#          - 1.6s : can capture at least one cardiac cycle in [40;240]bpm
l = int(framerate * 1.6) # 32

# Initialize
H = np.zeros(X.shape[0])

for n in range(X.shape[0]-l) :

    # Step1 :Spatial Averaging
    C = X[n:n+l-1,:].T # Sliding each window

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

from matplotlib import pyplot
pyplot.plot(range(H.shape[0]), H, 'g')
pyplot.title('Filtered green signal')
pyplot.show()






