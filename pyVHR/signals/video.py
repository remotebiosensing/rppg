import os
import re
import warnings

import cv2
import dlib
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import skvideo
import skvideo.io
from matplotlib import patches

from .pyramid import *
from ..utils import printutils
from ..utils.SkinDetect import SkinDetect


class Video:
    """
    Basic class for extracting ROIs from video frames
    """

    facePadding = 0.2  # dlib param for padding
    filenameCompressed = "croppedFaces.npz"  # filename to store on disk
    saveCropFaces = False  # enable the storage on disk of the cropped faces
    loadCropFaces = False  # enable the loading of cropped faces from disk

    def __init__(self, filename, verb=0):
        self.filename = filename
        self.faces = np.array([])  # empty array of cropped faces (RGB)
        self.masks = []
        self.processedFaces = np.array([])
        self.skinface = []
        self.faceSignal = []  # empty array of face signals (RGB) after roi/skin extraction
        self.dots = np.array([])
        self.generated_masks = np.array([])
        self.mask_result = np.array([])


        self.verb = verb
        self.cropSize = [150, 150]  # param for cropping
        self.typeROI = 'rect'  # type of rois between ['rect', 'skin']
        self.detector = 'mtcnn'
        self.time_vid_start = 0

        self.doEVM = False
        self.EVMalpha = 20
        self.EVMlevels = 3
        self.EVMlow = .8
        self.EVMhigh = 4

        self.rectCoords = [[0, 0, self.cropSize[0], self.cropSize[1]]]  # default 'rect' roi coordinates
        self.skinThresh_fix = [40, 80]  # default min values of Sauturation and Value (HSV) for 'skin' roi
        self.skinThresh_adapt = 0.2

    def getCroppedFaces(self, detector='mtcnn', extractor='skvideo', fps=30):
        """ Time is in seconds"""

        # -- check if cropped faces already exists on disk
        path, name = os.path.split(self.filename)
        filenamez = path + '/' + self.filenameCompressed

        self.detector = detector
        self.extractor = extractor

        # -- if compressed exists... load it
        if self.loadCropFaces and os.path.isfile(filenamez):
            self.cropped = True
            data = np.load(filenamez, allow_pickle=True)
            self.faces = data['a']
            self.numFrames = int(data['b'])
            self.frameRate = int(data['c'])
            self.height = int(data['d'])
            self.width = int(data['e'])
            self.duration = float(data['f'])
            self.codec = data['g']
            self.detector = data['h']
            self.extractor = data['i']
            self.cropSize = self.faces[0].shape

            if self.detector != detector:
                warnings.warn("\nWARNING!! Requested detector method is different from the saved one\n")

        # -- if compressed does not exist, load orig. video and extract faces
        else:
            self.cropped = False

            # if the video signal is stored in video container
            if os.path.isfile(self.filename):
                # -- metadata
                metadata = skvideo.io.ffprobe(self.filename)
                self.numFrames = int(eval(metadata["video"]["@nb_frames"]))
                self.height = int(eval(metadata["video"]["@height"]))
                self.width = int(eval(metadata["video"]["@width"]))
                self.frameRate = int(np.round(eval(metadata["video"]["@avg_frame_rate"])))
                self.duration = float(eval(metadata["video"]["@duration"]))
                self.codec = metadata["video"]["@codec_name"]
                # -- load video on a ndarray with skvideo or openCV
                video = None
                if extractor == 'opencv':
                    video = self.__opencvRead()
                else:
                    video = skvideo.io.vread(self.filename)

            # else if the video signal is stored as single frames
            else:  # elif os.path.isdir(self.filename):
                # -- load frames on a ndarray
                self.path = path
                video = self.__loadFrames()
                self.numFrames = len(video)
                self.height = video[0].shape[0]
                self.width = video[0].shape[1]
                self.frameRate = fps  ###### <<<<----- TO SET MANUALLY ####
                self.duration = self.numFrames / self.frameRate
                self.codec = 'raw'

            # -- extract faces and resize
            print('\n\n' + detector + '\n\n')
            self.__extractFace(video, method=detector)


            # -- store cropped faces on disk
            if self.saveCropFaces:
                np.savez_compressed(filenamez, a=self.faces,
                                    b=self.numFrames, c=self.frameRate,
                                    d=self.height, e=self.width,
                                    f=self.duration, g=self.codec,
                                    h=self.detector, i=self.extractor,
                                    )

        if '1' in str(self.verb):
            self.printVideoInfo()
            if not self.cropped:
                print('      Extracted faces: not found! Detecting...')
            else:
                print('      Extracted faces: found! Loading...')

    def setMask(self, typeROI='rect',
                rectCoords=None, rectRegions=None,
                skinThresh_fix=None, skinThresh_adapt=None):
        self.typeROI = typeROI
        if self.typeROI == 'rect':
            if rectCoords is not None:
                # List of rectangular ROIs: [[x0,y0,w0,h0],...,[xk,yk,wk,hk]]
                self.rectCoords = rectCoords
            elif rectRegions is not None:
                # List of rectangular regions: ['forehead', 'lcheek', 'rcheek', 'nose']
                self.rectCoords = self.__rectRegions2Coord(rectRegions)
        elif self.typeROI == 'skin_adapt' and skinThresh_adapt is not None:
            # Skin limits for HSV
            self.skinThresh_adapt = skinThresh_adapt
        elif self.typeROI == 'skin_fix' and skinThresh_fix is not None:
            # Skin limits for HSV
            self.skinThresh_fix = skinThresh_fix
        else:
            raise ValueError('Unrecognized type of ROI provided.')

    def extractSignal(self, frameSubset, count=None):
        if self.typeROI == 'rect':
            return self.__extractRectSignal(frameSubset)

        elif self.typeROI == 'skin_adapt' or self.typeROI == 'skin_fix' or self.typeROI =='adaption':
            return self.__extractSkinSignal(frameSubset, count)

    def setEVM(self, enable=True, alpha=20, levels=3, low=.8, high=4):
        """Eulerian Video Magnification"""

        # rawFaces = self.faces
        # gaussFaces = gaussian_video(rawFaces, levels=levels)
        # filtered = temporal_ideal_filter(gaussFaces, low, high, self.frameRate)
        # amplified = alpha * filtered
        # self.faces = reconstruct_video_g(amplified, rawFaces, levels=levels)
        self.doEVM = enable

        if enable is True:
            self.EVMalpha = alpha
            self.EVMlevels = levels
            self.EVMlow = low
            self.EVMhigh = high

    def applyEVM(self):
        vid_data = gaussian_video(self.faces, self.EVMlevels)
        vid_data = temporal_bandpass_filter(vid_data, self.frameRate,
                                            freq_min=self.EVMlow,
                                            freq_max=self.EVMhigh)
        vid_data *= self.EVMalpha
        self.processedFaces = combine_pyramid_and_save(vid_data,
                                                       self.faces,
                                                       enlarge_multiple=3,
                                                       fps=self.frameRate)

    def getMeanRGB(self):

        n_frames = len(self.faceSignal)
        n_roi = len(self.faceSignal[0])
        rgb = np.zeros([3, n_frames])

        for i in range(n_frames):
            mean_rgb = 0

            for roi in self.faceSignal[i]:
                idx = roi != 0
                idx2 = np.logical_and(np.logical_and(idx[:, :, 0], idx[:, :, 1]), idx[:, :, 2])
                roi = roi[idx2]
                if len(roi) == 0:
                    mean_rgb += 0
                else:
                     mean_rgb += np.mean(roi, axis=0)
                    # mean_rgb+=roi[roi != 0].mean()

            rgb[:, i] = mean_rgb / n_roi
        return rgb

    def printVideoInfo(self):
        print('\n   * Video filename: %s' % self.filename)
        print('         Total frames: %s' % self.numFrames)
        print('             Duration: %s (sec)' % np.round(self.duration, 2))
        print('           Frame rate: %s (fps)' % self.frameRate)
        print('                Codec: %s' % self.codec)

        printOK = 1
        try:
            f = self.numFrames
        except AttributeError:
            printOK = 0

        if printOK:
            print('           Num frames: %s' % self.numFrames)
            print('               Height: %s' % self.height)
            print('                Width: %s' % self.height)
            print('             Detector: %s' % self.detector)
            print('            Extractor: %s' % self.extractor)

    def printROIInfo(self):
        print('      ROI type: ' + self.typeROI)
        if self.typeROI == 'rect':
            print('   Rect coords: ' + str(self.rectCoords))
        elif self.typeROI == 'skin_fix':
            print('   Skin thresh: ' + str(self.skinThresh_fix))
        elif self.typeROI == 'skin_adapt':
            print('   Skin thresh: ' + str(self.skinThresh_adapt))

    def get_mask_num(self):
        return 31;

    def make_mask(self, bin_mask, dot):
        '''
        :param num:
        0 :  lower_cheek_left,          1 :  lower_cheek_right,          2 :  Malar_left
        3 :  Malar_right,               4 :  Marionette_Fold_left        5 :  Marionette_Fold_right
        6 :  chine                      7 :  Nasolabial_Fold_left        8 :  Nasolabial_Fold_right
        9 :  Lower_NasaL_Sidewall       10:  Upper_Lip_left              11:  Upper_Lip_right
        12:  Philtrum                   13:  Nasal_Tip                   14:  Lower_Nasal_Dorsum
        15: Lower_Nasal_Sidewall_left   16:  Lower_Nasal_Sidewall_right  17:  Mid_Nasal_Sidewall_left
        18: Mid_Nasal_Sidewall_right    19:  Upper_Nasal_Dorsum          20:  Glabella
        21: Lower_Lateral_Forehead_left 22: Lower_Lateral_Forehead_right 23: temporal_lobe_left
        24: temporal_lobe_right         25: eye_left                     26: eye_right
        27: Lower_Medial_Forehead       28: Upper_Lateral_Forehead_left  29: Upper_Lateral_Forehead_right
        30: Upper_Medial_Forehead
        :param dot:
        :return:
        '''
        mask_list = []
        for (idx, bin) in enumerate(bin_mask):
            if bin == '1':
                if idx == 0:  # lower_cheek_left
                    mask_list.append(np.array(
                        [avg(dot[132], dot[123]), dot[132], dot[215], dot[172], dot[136],
                         dot[169], dot[210], avg(dot[212], dot[202]), dot[57],
                         avg(dot[61], dot[186]),
                         dot[92], dot[206], dot[205], avg(dot[50], dot[147])]
                    ))
                elif idx == 1:
                    mask_list.append(np.array(
                        [avg(dot[361], dot[352]), dot[361], dot[435], dot[397], dot[365],
                         dot[394], dot[430], avg(dot[273], dot[287]), dot[287],
                         avg(dot[391], dot[410]),
                         dot[322], dot[426], dot[425], avg(dot[411], dot[280])]))
                elif idx == 2:
                    mask_list.append(np.array(
                        [avg(dot[116], dot[93]), dot[116], dot[117], dot[118], dot[100],
                         avg(dot[126], dot[142]),
                         avg(dot[209], dot[142]), dot[49], dot[203], dot[205], dot[123]]
                    ))
                elif idx == 3:
                    mask_list.append(
                        np.array([avg(dot[345], dot[323]), dot[345], dot[346], dot[347], dot[329],
                                  avg(dot[420], dot[371]),
                                  avg(dot[371], dot[360]), dot[429], dot[423], dot[425], dot[352]]))
                elif idx == 4:
                    mask_list.append(np.array(
                        [dot[61], dot[43], dot[204], dot[32], dot[171], dot[148], dot[176],
                         dot[149],
                         dot[150],
                         dot[169], dot[210], avg(dot[212], dot[202]),
                         avg(dot[57], avg(dot[57], dot[43]))]))
                elif idx == 5:
                    mask_list.append( np.array(
                [dot[291], dot[273], dot[424], dot[262], dot[396], dot[377], dot[400], dot[378],
                 dot[379],
                 dot[394], dot[430], avg(dot[432], dot[422]),
                 avg(dot[287], avg(dot[287], dot[273]))]))
                elif idx==6:
                    mask_list.append(np.array(
                [dot[204], avg(dot[106], dot[91]), avg(dot[182], dot[181]), avg(dot[83], dot[84]),
                 avg(dot[18], dot[17]), avg(dot[314], dot[313]), avg(dot[405], dot[406]),
                 avg(dot[321], dot[335]), dot[424], dot[262], dot[396], dot[377], dot[152],
                 dot[148],
                 dot[171], dot[32]]))
                elif idx == 7:
                    mask_list.append( np.array(
                [dot[61],dot[92],avg(dot[206],dot[203]),dot[102],dot[48],dot[64],dot[98],dot[60],dot[165]]))
                elif idx == 8:
                    mask_list.append(np.array(
                [dot[291], dot[322], avg(dot[426],dot[423]), dot[331],dot[294],dot[327],dot[290],dot[391]]))
                elif idx == 9:
                    mask_list.append(np.array(
                [dot[240], dot[64], dot[48], dot[131], avg(dot[134], avg(dot[134], dot[220])),
                 dot[45], avg(dot[4], avg(dot[4], dot[1])),
                 dot[275], avg(dot[363], avg(dot[363], dot[440])), dot[360], dot[278], dot[294],
                 dot[460], dot[305], dot[309], dot[438],
                 avg(dot[457], dot[275]), avg(dot[274], dot[275]), dot[274],
                 dot[458], dot[461], dot[462], dot[370],
                 dot[94], dot[141],
                 dot[242], dot[241], dot[238], dot[44],
                 avg(dot[45], dot[44]),
                 avg(dot[218], dot[45]), dot[218], dot[79], dot[75]]))
                elif idx == 10:
                    mask_list.append(np.array(
                [dot[60], dot[165], dot[61], dot[40], dot[39], dot[37], dot[97], dot[99]]))
                elif idx == 11:
                    mask_list.append(np.array(
                [dot[290], dot[391], dot[291], dot[270], dot[269], dot[267], dot[326],
                 dot[328]]))
                elif idx == 12:
                    mask_list.append(np.array(
                [dot[2], avg(dot[242], dot[97]), avg(dot[37], avg(dot[242], dot[37])), dot[0],
                 avg(dot[267], avg(dot[267], dot[462])), avg(dot[462], dot[326])]))
                elif idx == 13:
                    mask_list.append(np.array([dot[4], avg(dot[45], avg(dot[45], dot[51])), dot[134],
                                  dot[51], dot[5], dot[281], dot[363],
                                  avg(dot[275], avg(dot[275], dot[281]))]))
                elif idx == 14:
                    mask_list.append( np.array(
                [dot[195], dot[248], dot[456], avg(dot[281], avg(dot[281], dot[248]))
                    , dot[5], avg(dot[51], avg(dot[51], dot[3])), dot[236], dot[3]]))
                elif idx == 15:
                    mask_list.append(
                        np.array([dot[236], avg(dot[236], dot[134]),
                                  avg(dot[198], dot[131]), avg(dot[126], dot[209]),
                                  avg(dot[217], dot[198])])
                    )
                elif idx == 16:
                    mask_list.append(
                        np.array([dot[456], avg(dot[456], dot[363]),
                                  avg(dot[420], dot[360]), avg(dot[355], dot[429]),
                                  avg(dot[437], dot[420])])
                    )
                elif idx == 17:
                    mask_list.append(
                        np.array([dot[196], avg(dot[114], dot[217]), dot[236]])
                    )
                elif idx == 18:
                    mask_list.append(
                        np.array([dot[419], avg(dot[343], dot[437]), dot[456]])
                    )
                elif idx == 19:
                    mask_list.append(
                        np.array(
                            [avg(dot[114], dot[196]), dot[197], avg(dot[343], dot[419]), dot[351],
                             dot[417],
                             dot[285], dot[8],
                             dot[55], dot[193], dot[122]])
                    )
                elif idx == 20:
                    mask_list.append(
                        np.array(
                            [avg(dot[55], avg(dot[55], dot[8])), dot[8],
                             avg(dot[285], avg(dot[8], dot[285])),
                             dot[336],
                             avg(dot[337], avg(dot[337], dot[336])),
                             avg(dot[151], avg(dot[151], dot[9])),
                             avg(dot[108], avg(dot[108], dot[107])), dot[107]])
                    )
                elif idx == 21:
                    mask_list.append(
                        np.array(
                            [avg(dot[108], avg(dot[109], dot[69])),
                             avg(dot[104], avg(dot[68], dot[104])),
                             dot[105], dot[107]])
                    )
                elif idx == 22:
                    mask_list.append(
                        np.array(
                            [avg(dot[337], avg(dot[338], dot[299])),
                             avg(dot[333], avg(dot[298], dot[333])),
                             dot[334], dot[336]])
                    )
                elif idx == 23:
                    mask_list.append(
                        np.array(
                            [avg(dot[54], dot[103]), avg(dot[104], dot[68]), avg(dot[63], dot[105]),
                             dot[70],
                             dot[156], dot[143], dot[116], dot[234], dot[127],
                             dot[162], dot[21]])
                    )
                elif idx == 24:
                    mask_list.append(
                        np.array(
                            [avg(dot[332], dot[284]), avg(dot[298], dot[333]),
                             avg(dot[334], dot[293]),
                             dot[300], dot[383], dot[372], dot[345], dot[454], dot[356],
                             dot[389], dot[251]])
                    )
                elif idx == 25:
                    mask_list.append(
                        np.array(
                            [dot[53], dot[46], dot[156], dot[143], dot[111], dot[117], dot[118],
                             dot[119],
                             dot[120], dot[121],
                             dot[128], dot[245], dot[193], avg(dot[221],dot[55]), avg(dot[222],avg(dot[222],dot[65])), dot[52]])
                    )
                elif idx == 26:
                    mask_list.append(
                        np.array(
                            [dot[283], dot[276], dot[383], dot[372], dot[340], dot[346], dot[347],
                             dot[348], dot[349], dot[350], dot[357], dot[465], dot[417], dot[441], avg(dot[442],avg(dot[442],dot[295])), avg(dot[285],dot[441]), dot[282]  ]))
                elif idx == 27:
                    mask_list.append(
                        np.array(
                            [
                                avg(dot[108], avg(dot[108], dot[109])),
                                avg(dot[151], avg(dot[10], avg(dot[151], avg(dot[151], dot[10])))),
                                avg(dot[337], avg(dot[337], dot[338])),
                                avg(dot[337], avg(dot[338], dot[299])),
                                avg(dot[337], avg(dot[337], dot[336])),
                                avg(dot[151], avg(dot[151], dot[9])),
                                avg(dot[108], avg(dot[108], dot[107])),
                                avg(dot[108], avg(dot[109], dot[69]))
                            ]
                        )
                    )
                elif idx == 28:
                    mask_list.append(
                        np.array(
                            [dot[103], dot[67], avg(dot[109], avg(dot[109], dot[67])),
                             avg(dot[108], avg(dot[108], dot[69])), dot[69], dot[104]])
                    )
                elif idx == 29:
                    mask_list.append(
                        np.array(
                            [dot[332], dot[297], avg(dot[338], avg(dot[338], dot[297])),
                             avg(dot[337], avg(dot[337], dot[299])), dot[299], dot[333]])
                    )
                elif idx == 30:
                    mask_list.append(
                        np.array(
                            [
                                avg(dot[108], avg(dot[109], dot[69])),
                                avg(dot[108], avg(dot[108], dot[109])),
                                avg(dot[151], avg(dot[10], avg(dot[151], avg(dot[151], dot[10])))),
                                avg(dot[337], avg(dot[337], dot[338])),
                                avg(dot[337], avg(dot[338], dot[299])), dot[338], dot[10], dot[109]
                            ]
                    ))


        self.masks.extend([mask_list])

    def generate_maks(self,seq):
        #img = self.faces[seq]
        img = np.zeros((self.cropSize[0],self.cropSize[1],3),np.uint8)
        # img = cv2.cvtColor(self.faces[seq],cv2.COLOR_BGR2GRAY)
        for (idx,mask) in enumerate(self.masks[seq]):
            # print(mask)
            img = cv2.fillConvexPoly(img, mask.astype(int), color=(255, 255, 255))
            # img = cv2.putText(img=img,text= str(idx),org = np.mean(mask.astype(int),axis=0).astype(int),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.4,color=(0,0,0))
            # img = cv2.polylines(img,[mask.astype(int)],isClosed=True,color=(0,0,255))

        self.generated_masks[seq,:,:] = img
        # self.faces[seq] = cv2.cvtColor(self.faces[seq], cv2.COLOR_BGR2RGB)
        rst = cv2.bitwise_and(self.faces[seq],img)
        self.mask_result[seq,:,:] = rst
        self.skinface.append([rst])
        # count = 0
        # for i in rst:
        #     for j in i:
        #         if j.mean() != 0:
        #             count +=1
        # return count



    def showVideo(self):
        # from ipywidgets import interact
        # import ipywidgets as widgets

        n = self.numFrames

        def view_image(frame):

            idx = frame - 1

            if self.processedFaces.size == 0:
                face = self.faces[idx]
            else:
                face = self.processedFaces[idx]

            if self.typeROI == 'rect':
                plt.imshow(face, interpolation='nearest')

                ax = plt.gca()

                for coord in self.rectCoords:
                    rect = patches.Rectangle((coord[0], coord[1]),
                                             coord[2], coord[3], linewidth=1, edgecolor='y', facecolor='none')
                    ax.add_patch(rect)

            elif self.typeROI == 'skin_fix':
                lower = np.array([0, self.skinThresh_fix[0], self.skinThresh_fix[1]], dtype="uint8")
                upper = np.array([20, 255, 255], dtype="uint8")
                converted = cv2.cvtColor(face, cv2.COLOR_RGB2HSV)
                skinMask = cv2.inRange(converted, lower, upper)
                skinFace = cv2.bitwise_and(face, face, mask=skinMask)
                plt.imshow(skinFace, interpolation='nearest')

            elif self.typeROI == 'skin_adapt':
                sd = SkinDetect(strength=self.skinThresh_adapt)
                sd.compute_stats(face)
                skinFace = sd.get_skin(face, filt_kern_size=7, verbose=False, plot=False)
                plt.imshow(skinFace, interpolation='nearest')

        # interact(view_image, frame=widgets.IntSlider(min=1, max=n, step=1, value=1))

    def __opencvRead(self):
        vid = cv2.VideoCapture(self.filename)
        frames = []
        retval, frame = vid.read()
        while retval == True:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            retval, frame = vid.read()
        vid.release()
        return np.asarray(frames)

    def __extractRectSignal(self, frameSubset):
        """ Extract R,G,B values on all ROIs of a frame subset """

        assert self.processedFaces.size > 0, "Faces are not processed yet! Please call runOffline first"

        self.faceSignal = []

        i = 0
        for r in frameSubset:
            face = self.processedFaces[r]
            H = face.shape[0]
            W = face.shape[1]

            # take frame-level rois
            rois = []
            for roi in self.rectCoords:
                x = roi[0]
                y = roi[1]
                w = min(x + roi[2], W)
                h = min(y + roi[3], H)
                rois.append(face[y:h, x:w, :])

            # take all rois of the frame
            self.faceSignal.append(rois)
            i += 1

    def __extractSkinSignal(self, frameSubset, count=None, frameByframe=False):
        """ Extract R,G,B values from skin-based roi of a frame subset """

        assert self.processedFaces.size > 0, "Faces are not processed yet! Please call runOffline first"

        self.faceSignal = []

        cp = self.cropSize
        skinFace = np.zeros([cp[0], cp[1], 3], dtype='uint8')

        # -- loop on frames
        for i, r in enumerate(frameSubset):
            face = self.processedFaces[r]

            if self.typeROI == 'skin_fix':
                assert len(self.skinThresh_fix) == 2, "Please provide 2 values for Fixed Skin Detector"
                lower = np.array([0, self.skinThresh_fix[0], self.skinThresh_fix[1]], dtype="uint8")
                upper = np.array([20, 255, 255], dtype="uint8")
                converted = cv2.cvtColor(face, cv2.COLOR_RGB2HSV)
                skinMask = cv2.inRange(converted, lower, upper)
                skinFace = cv2.bitwise_and(face, face, mask=skinMask)
                self.faceSignal.append([skinFace])

            elif self.typeROI == 'skin_adapt':
                if count == 0 and i == 0:
                    self.sd = SkinDetect(strength=self.skinThresh_adapt)
                    self.sd.compute_stats(face)

                if frameByframe and i > 0:
                    self.sd.compute_stats(face)

                skinFace = self.sd.get_skin(face, filt_kern_size=0, verbose=False, plot=False)

                self.faceSignal.append([skinFace])
            elif self.typeROI == 'adaption':
                self.faceSignal.append(self.skinface[r])

    def __extractFace(self, video, method, t_downsample_rate=2):

        # -- save on GPU
        # self.facesGPU = cp.asarray(self.faces)  # move the data to the current device.

        if method == 'media-pipe':
            detector = FaceMeshDetector(maxFaces=2)

            self.mask_result = np.zeros([self.numFrames, self.cropSize[0], self.cropSize[1], 3], dtype='uint8')
            self.generated_masks = np.zeros([self.numFrames, self.cropSize[0], self.cropSize[1],3], dtype='uint8')
            self.faces = np.zeros([self.numFrames, self.cropSize[0], self.cropSize[1], 3], dtype='uint8')
            self.dots = np.zeros([self.numFrames,468,2])

            for i in range(self.numFrames):
                frame = video[i, :, :, :]
                self.numFaces = 0
                _, dot = detector.findFaceMesh(frame)
                # self.dots[i,:,:] = dot[0]
                if len(dot) > 0:
                    self.numFaces += 1
                    x_min = min(np.array(dot[0][:]).T[0])
                    y_min = min(np.array(dot[0][:]).T[1])
                    x_max = max(np.array(dot[0][:]).T[0])
                    y_max = max(np.array(dot[0][:]).T[1])
                    f = frame[y_min:y_max, x_min:x_max]
                    f = cv2.resize(f, dsize=(self.cropSize[0], self.cropSize[1]))
                    _,dot = detector.findFaceMesh(f)
                    if len(dot) == 0:
                        return
                    # f = cv2.cvtColor(f,cv2.COLOR_BGR2RGB)
                    # for o in dot[0] :
                    #     f = cv2.circle(f,o,1,(0,0,255))
                    self.dots[i, :, :] = dot[0]
                    self.faces[i, :, :, :] = f.astype('uint8')

        elif method == 'dlib':
            # -- dlib detector
            detector = dlib.get_frontal_face_detector()
            if os.path.exists("resources/shape_predictor_68_face_landmarks.dat"):
                file_predict = "resources/shape_predictor_68_face_landmarks.dat"
            elif os.path.exists("../resources/shape_predictor_68_face_landmarks.dat"):
                file_predict = "../resources/shape_predictor_68_face_landmarks.dat"
            predictor = dlib.shape_predictor(file_predict)
            self.faces = np.zeros([self.numFrames, self.cropSize[0], self.cropSize[1], 3],
                                  dtype='uint8')

            # -- loop on frames
            cp = self.cropSize
            self.faces = np.zeros([self.numFrames, cp[0], cp[1], 3], dtype='uint8')
            for i in range(self.numFrames):
                frame = video[i, :, :, :]
                # -- Detect face using dlib
                self.numFaces = 0
                facesRect = detector(frame, 0)
                if len(facesRect) > 0:
                    # -- process only the first face
                    self.numFaces += 1
                    rect = facesRect[0]
                    x0 = rect.left()
                    y0 = rect.top()
                    w = rect.width()
                    h = rect.height()

                    # -- extract cropped faces
                    shape = predictor(frame, rect)
                    f = dlib.get_face_chip(frame, shape, size=self.cropSize[0], padding=self.facePadding)
                    self.faces[i, :, :, :] = f.astype('uint8')

                if self.verb:
                    printutils.printProgressBar(i, self.numFrames, prefix='Processing:', suffix='Complete', length=50)

                else:
                    print("No face detected at frame %s", i)

        elif method == 'mtcnn_kalman':
            # mtcnn detector
            from mtcnn import MTCNN
            detector = MTCNN()

            h0 = None
            w0 = None
            crop = np.zeros([2, 2, 2])
            skipped_frames = 0

            while crop.shape[:2] != (h0, w0):
                if skipped_frames > 0:
                    print("\nWARNING! Strange Face Crop... Skipping frame " + str(skipped_frames) + '...')
                frame = video[skipped_frames, :, :, :]
                detection = detector.detect_faces(frame)

                if len(detection) > 1:
                    areas = []
                    for det in detection:
                        areas.append(det['box'][2] * det['box'][3])
                    areas = np.array(areas)
                    ia = np.argsort(areas)
                    [x0, y0, w0, h0] = detection[ia[-1]]['box']
                else:
                    [x0, y0, w0, h0] = detection[0]['box']

                w0 = 2 * (int(w0 / 2))
                h0 = 2 * (int(h0 / 2))
                # Cropping face
                crop = frame[y0:y0 + h0, x0:x0 + w0, :]

                skipped_frames += 1

            self.cropSize = crop.shape[:2]

            if skipped_frames > 1:
                self.numFrames = self.numFrames - skipped_frames
                new_time_vid_start = skipped_frames / self.frameRate

                if new_time_vid_start > self.time_vid_start:
                    self.time_vid_start = new_time_vid_start
                    print("\tVideo now starts at " + str(self.time_vid_start) + " seconds\n")

            self.faces = np.zeros([self.numFrames, self.cropSize[0], self.cropSize[1], 3], dtype='uint8')
            self.faces[0, :, :, :] = crop

            # set the initial tracking window
            state = np.array([int(x0 + w0 / 2), int(y0 + h0 / 2), 0, 0], dtype='float64')  # initial position

            # Setting up Kalman Filter
            kalman = cv2.KalmanFilter(4, 2, 0)
            kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
                                                [0., 1., 0., .1],
                                                [0., 0., 1., 0.],
                                                [0., 0., 0., 1.]])
            kalman.measurementMatrix = 1. * np.eye(2, 4)
            kalman.processNoiseCov = 1e-5 * np.eye(4, 4)
            kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
            kalman.errorCovPost = 1e-1 * np.eye(4, 4)
            kalman.statePost = state
            measurement = np.array([int(x0 + w0 / 2), int(y0 + h0 / 2)], dtype='float64')

            for i in range(skipped_frames, self.numFrames):
                frame = video[i, :, :, :]

                if i % t_downsample_rate == 0:
                    detection = detector.detect_faces(frame)
                    if len(detection) != 0:
                        areas = []
                        if len(detection) > 1:
                            for det in detection:
                                areas.append(det['box'][2] * det['box'][3])
                            areas = np.array(areas)
                            ia = np.argsort(areas)

                            [x0, y0, w, h] = detection[ia[-1]]['box']
                        else:
                            [x0, y0, w, h] = detection[0]['box']

                        not_found = False
                    else:
                        not_found = True

                prediction = kalman.predict()  # prediction

                if i % t_downsample_rate == 0 and not not_found:
                    measurement = np.array([x0 + w / 2, y0 + h / 2], dtype='float64')
                    posterior = kalman.correct(measurement)
                    [cx0, cy0, wn, hn] = posterior.astype(int)
                else:
                    [cx0, cy0, wn, hn] = prediction.astype(int)

                # Cropping with new bounding box
                crop = frame[int(cy0 - h0 / 2):int(cy0 + h0 / 2), int(cx0 - w0 / 2):int(cx0 + w0 / 2), :]

                if crop.shape[:2] != self.faces.shape[1:3]:
                    print("WARNING! Strange face crop: video frame " + str(
                        i) + " probably does not contain the whole face... Reshaping Crop\n")
                    crop = cv2.resize(crop, (self.faces.shape[2], self.faces.shape[1]))

                self.faces[i, :, :, :] = crop.astype('uint8')

        elif method == 'mtcnn':
            # mtcnn detector
            from mtcnn import MTCNN
            # from utils.FaceAligner import FaceAligner
            detector = MTCNN()

            print("\nPerforming face detection...")

            h0 = None
            w0 = None
            crop = np.zeros([2, 2, 2])
            skipped_frames = 0

            while crop.shape[:2] != (h0, w0):
                if skipped_frames > 0:
                    print("\nWARNING! Strange Face Crop... Skipping frame " + str(skipped_frames) + '...')
                frame = video[skipped_frames, :, :, :]
                detection = detector.detect_faces(frame)

                if len(detection) == 0:
                    skipped_frames += 1
                    continue

                if len(detection) > 1:
                    areas = []
                    for det in detection:
                        areas.append(det['box'][2] * det['box'][3])
                    areas = np.array(areas)
                    ia = np.argsort(areas)
                    [x0, y0, w0, h0] = detection[ia[-1]]['box']
                    nose = detection[ia[-1]]['keypoints']['nose']
                    r_eye = detection[ia[-1]]['keypoints']['right_eye']
                    l_eye = detection[ia[-1]]['keypoints']['left_eye']
                else:
                    [x0, y0, w0, h0] = detection[0]['box']
                    nose = detection[0]['keypoints']['nose']
                    r_eye = detection[0]['keypoints']['right_eye']
                    l_eye = detection[0]['keypoints']['left_eye']

                w0 = 2 * (int(w0 / 2))
                h0 = 2 * (int(h0 / 2))
                barycenter = (np.array(nose) + np.array(r_eye) + np.array(l_eye)) / 3.
                cy0 = barycenter[1]
                cx0 = barycenter[0]
                # Cropping face
                crop = frame[int(cy0 - h0 / 2):int(cy0 + h0 / 2), int(cx0 - w0 / 2):int(cx0 + w0 / 2), :]

                skipped_frames += 1

            # fa = FaceAligner(desiredLeftEye=(0.3, 0.3),desiredFaceWidth=w0, desiredFaceHeight=h0)
            # crop_align = fa.align(frame, r_eye, l_eye)

            self.cropSize = crop.shape[:2]

            if skipped_frames > 1:
                self.numFrames = self.numFrames - skipped_frames
                new_time_vid_start = skipped_frames / self.frameRate

                if new_time_vid_start > self.time_vid_start:
                    self.time_vid_start = new_time_vid_start
                    print("\tVideo now starts at " + str(self.time_vid_start) + " seconds\n")

            self.faces = np.zeros([self.numFrames, self.cropSize[0], self.cropSize[1], 3], dtype='uint8')
            self.faces[0, :, :, :] = crop

            old_detection = detection
            for i in range(skipped_frames, self.numFrames):
                # print('\tFrame ' + str(i) + ' of ' + str(self.numFrames))
                frame = video[i, :, :, :]

                new_detection = detector.detect_faces(frame)
                areas = []

                if len(new_detection) == 0:
                    new_detection = old_detection

                if len(new_detection) > 1:
                    for det in new_detection:
                        areas.append(det['box'][2] * det['box'][3])
                    areas = np.array(areas)
                    ia = np.argsort(areas)

                    [x0, y0, w, h] = new_detection[ia[-1]]['box']
                    nose = new_detection[ia[-1]]['keypoints']['nose']
                    r_eye = new_detection[ia[-1]]['keypoints']['right_eye']
                    l_eye = new_detection[ia[-1]]['keypoints']['left_eye']
                else:
                    [x0, y0, w, h] = new_detection[0]['box']
                    nose = new_detection[0]['keypoints']['nose']
                    r_eye = new_detection[0]['keypoints']['right_eye']
                    l_eye = new_detection[0]['keypoints']['left_eye']

                barycenter = (np.array(nose) + np.array(r_eye) + np.array(l_eye)) / 3.
                cy0 = barycenter[1]
                cx0 = barycenter[0]
                # Cropping with new bounding box
                crop = frame[int(cy0 - h0 / 2):int(cy0 + h0 / 2), int(cx0 - w0 / 2):int(cx0 + w0 / 2), :]

                if crop.shape[:2] != self.faces.shape[1:3]:
                    print("WARNING! Strange face crop: video frame " + str(
                        i) + " probably does not contain the whole face... Reshaping Crop\n")
                    crop = cv2.resize(crop, (self.faces.shape[2], self.faces.shape[1]))

                self.faces[i, :, :, :] = crop.astype('uint8')
                old_detection = new_detection

                # if self.verb:
                printutils.printProgressBar(i, self.numFrames, prefix='Processing:', suffix='Complete', length=50)
        else:

            raise ValueError('Unrecognized Face detection method. Please use "dlib" or "mtcnn"')

    def __rectRegions2Coord(self, rectRegions):

        # regions 'forehead'
        #         'lcheek'
        #         'rcheek'
        #         'nose'
        assert len(self.faces) > 0, "Faces not found, please run getCroppedFaces first!"

        w = self.faces[0].shape[1]
        h = self.faces[0].shape[0]

        coords = []

        for roi in rectRegions:
            if roi == 'forehead':
                if self.detector == 'dlib':
                    x_f = int(w * .34)
                    y_f = int(h * .05)
                    w_f = int(w * .32)
                    h_f = int(h * .05)

                elif (self.detector == 'mtcnn') or (self.detector == 'mtcnn_kalman'):
                    x_f = int(w * .20)
                    y_f = int(h * .10)
                    w_f = int(w * .60)
                    h_f = int(h * .12)

                coords.append([x_f, y_f, w_f, h_f])

            elif roi == 'lcheek':
                if self.detector == 'dlib':
                    x_c = int(w * .22)
                    y_c = int(h * .40)
                    w_c = int(w * .14)
                    h_c = int(h * .11)

                elif (self.detector == 'mtcnn') or (self.detector == 'mtcnn_kalman'):
                    x_c = int(w * .15)
                    y_c = int(h * .54)
                    w_c = int(w * .15)
                    h_c = int(h * .11)

                coords.append([x_c, y_c, w_c, h_c])

            elif roi == 'rcheek':
                if self.detector == 'dlib':
                    x_c = int(w * .64)
                    y_c = int(h * .40)
                    w_c = int(w * .14)
                    h_c = int(h * .11)

                elif (self.detector == 'mtcnn') or (self.detector == 'mtcnn_kalman'):
                    x_c = int(w * .70)
                    y_c = int(h * .54)
                    w_c = int(w * .15)
                    h_c = int(h * .11)

                coords.append([x_c, y_c, w_c, h_c])

            elif roi == 'nose':
                if self.detector == 'dlib':
                    x_c = int(w * .40)
                    y_c = int(h * .35)
                    w_c = int(w * .20)
                    h_c = int(h * .05)

                elif (self.detector == 'mtcnn') or (self.detector == 'mtcnn_kalman'):
                    x_c = int(w * .35)
                    y_c = int(h * .50)
                    w_c = int(w * .30)
                    h_c = int(h * .08)

                coords.append([x_c, y_c, w_c, h_c])

            else:
                raise ValueError('Unrecognized rect region name.')

        return coords

    def __sort_nicely(self, l):
        """ Sort the given list in the way that humans expect. 
        """
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        l.sort(key=alphanum_key)
        return l

    def __loadFrames(self):

        # -- delete the compressed if exists
        cmpFile = os.path.join(self.path, self.filenameCompressed)
        if os.path.exists(cmpFile):
            os.remove(cmpFile)

        # -- get filenames within dir
        f_names = self.__sort_nicely(os.listdir(self.path))
        frames = []
        for n in range(len(f_names)):
            filename = os.path.join(self.path, f_names[n])
            frames.append(cv2.imread(filename)[:, :, ::-1])

        frames = np.array(frames)
        return frames


class FaceMeshDetector:

    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):

        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils

        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection()

        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces,
                                                 self.minDetectionCon, self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(img)
        #self.faces = self.faceDetection.process(img)

        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:

                face = []
                for id, lm in enumerate(faceLms.landmark):
                    # print(lm)
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    # cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN,
                    #           0.7, (0, 255, 0), 1)

                    # print(id,x,y)
                    face.append([x, y])
                faces.append(face)
        return img, faces


def avg(a, b):
    return [(int)((x + y) / 2) for x, y in zip(a, b)]
