import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import cv2
import heartpy.peakdetection as hp_peak
from heartpy.datautils import rolling_mean
from heartpy.filtering import filter_signal
from scipy.signal import stft


def getPPG_UBFC_rPPG(dataPath):
    return np.loadtxt(dataPath + '/ground_truth.txt')[0]


def getHR_UBFC_rPPG(dataPath):
    return np.loadtxt(dataPath + '/ground_truth.txt')[1]


def getSampleRate_UBFC_rPPG(dataPath):
    VidObj = cv2.VideoCapture(dataPath + '/vid.avi')
    vid_len = int(VidObj.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = VidObj.get(cv2.CAP_PROP_FPS)
    VidObj.release()

    PPG = getPPG_UBFC_rPPG(dataPath)
    duration = vid_len / fps
    return len(PPG) / duration


def makeSignalDict_UBFC_rPPG(dataPath):
    PPG = getPPG_UBFC_rPPG(dataPath)
    HR = getHR_UBFC_rPPG(dataPath)
    sampleRate = getSampleRate_UBFC_rPPG(dataPath)
    duration = len(PPG)/sampleRate

    PPG_scaled = filter_signal(PPG, cutoff=3, sample_rate=sampleRate, order=2, filtertype='lowpass')
    PPG_scaled = 2 * (PPG_scaled - np.min(PPG_scaled)) / (np.max(PPG_scaled) - np.min(PPG_scaled)) - 1

    signalDict = {'PPG': PPG, 'PPG_scaled': PPG_scaled, 'HR': HR, 'SampleRate': sampleRate, 'Duration': duration}
    return signalDict


def getSubjectNumList_UBFC_rPPG(rootPath):
    return sorted(list(map(lambda x: int(x[len(rootPath + '/subject'):]), glob(rootPath + '/subject*'))))


def getSubjectDict_UBFC_rPPG(rootPath, subjectList=None):
    # subjectList(sNum): Range 1~49, Length 40, list(int)

    if subjectList is None:
        subjectList = getSubjectNumList_UBFC_rPPG(rootPath)

    subjectDict = {}
    for sNum in subjectList:
        if os.path.isdir(rootPath + f'/subject{sNum}'):
            subjectDict[f'Subject{sNum}'] = makeSignalDict_UBFC_rPPG(rootPath + f'/subject{sNum}')
        else:
            print('Not Found: ' + f'Subject{sNum}')
    return subjectDict


def getPowerSpectrum(signalDict):
    fourier_transform = np.fft.rfft(signalDict['PPG'])
    abs_fourier_transform = np.abs(fourier_transform)
    powerSpectrumPoints = np.square(abs_fourier_transform)
    freqIdxPoints = np.linspace(0, signalDict['SampleRate'] / 2, len(powerSpectrumPoints))

    return freqIdxPoints, powerSpectrumPoints


def getPeaks(signalDict, windowSize=5, rolling_mean_windowSize=1.2):
    peaks = np.array([], dtype=int)
    win = int(signalDict['SampleRate'] * windowSize)
    for i in range(0, len(signalDict['PPG_scaled']), win):
        try:
            roll_mean = rolling_mean(signalDict['PPG_scaled'][i:i + win], rolling_mean_windowSize, signalDict['SampleRate'])
            peak_heartpy = hp_peak.detect_peaks(signalDict['PPG_scaled'][i:i+win], roll_mean,
                                                ma_perc=20, sample_rate=signalDict['SampleRate'])
            peaks = np.append(peaks, np.array(peak_heartpy['peaklist'], dtype=int) + i)
        except ValueError:
            pass
        except IndexError:
            pass
    return peaks


def fitPeaks_v1(signalDict, peaksIdx):
    # Assume
    # 1) getPeak에서 얻은 Point가 적어도 인접 노치보다 큰 경우
    # 2) getPeak에서 얻은 Point 수 <= 실제 Peak 수
    reference = signalDict['PPG']
    output = peaksIdx.copy()

    for i in range(len(peaksIdx)):
        tempPoint = peaksIdx[i]
        if (tempPoint == 0) | (tempPoint == len(reference) - 1):
            continue

        if (reference[tempPoint - 1] < reference[tempPoint]) & (
                reference[tempPoint] > reference[tempPoint + 1]):  # Peak
            continue
        elif (reference[tempPoint - 1] > reference[tempPoint]) & (
                reference[tempPoint] > reference[tempPoint + 1]):  # Direction (<-)
            direction = -1
        elif (reference[tempPoint - 1] < reference[tempPoint]) & (
                reference[tempPoint] < reference[tempPoint + 1]):  # Direction (->)
            direction = 1
        else: # (_/ or \_ or \/ ..)
            continue

        while (tempPoint > 0) & (tempPoint < len(reference) - 1):
            if direction == -1:
                tempPoint -= 1
            elif direction == 1:
                tempPoint += 1

            if (i != 0) & (i != len(peaksIdx) - 1):
                if (tempPoint == peaksIdx[i + 1]) | (tempPoint == peaksIdx[i - 1]):
                    direction = -direction
                    continue

            if (tempPoint > 0) & (tempPoint < len(reference) - 1):
                if (reference[tempPoint - 1] <= reference[tempPoint]) & (
                        reference[tempPoint] >= reference[tempPoint + 1]):  # Peak
                    output[i] = tempPoint
                    break
    return output


def getOutlierIdx(dataPoints, weight=1.5):
    quantile_25 = np.percentile(dataPoints, 25)
    quantile_75 = np.percentile(dataPoints, 75)

    IQR = quantile_75 - quantile_25
    IQR_weight = IQR * weight

    lowest = quantile_25 - IQR_weight
    highest = quantile_75 + IQR_weight

    return np.where((dataPoints > highest) | (dataPoints < lowest))[0]


def getBPM_from_GT(signalDict):
    timeIdxPoints = np.linspace(0, signalDict['Duration'], len(signalDict['HR']))
    gt_bpmPoints = signalDict['HR']
    return timeIdxPoints, gt_bpmPoints


def getBPM_by_Peaks(signalDict):
    peaksIdx_raw = getPeaks(signalDict)
    peaksIdx = fitPeaks_v1(signalDict, peaksIdx_raw)
    timeIdx = (peaksIdx/signalDict['SampleRate'])[:-1]

    diffPeak = np.diff(peaksIdx)
    bpmPoints = (signalDict['SampleRate']/diffPeak)*60

    outlierIdx = getOutlierIdx(bpmPoints)
    outlierValues = bpmPoints[outlierIdx]

    timeIdxPoints = np.delete(timeIdx, outlierIdx)
    bpmPoints = np.delete(bpmPoints, outlierIdx)

    return timeIdxPoints, bpmPoints, timeIdx[outlierIdx], outlierValues


class BVPsignal:
    """
    Manage (multi-channel, row-wise) BVP signals, and transforms them in BPMs.
    """
    #nFFT = 2048  # freq. resolution for STFTs
    step = 1       # step in seconds

    def __init__(self, data, fs, startTime=0, minHz=0.75, maxHz=4.):
        if len(data.shape) == 1:
            self.data = data.reshape(1, -1)  # 2D array raw-wise
        else:
            self.data = data
        self.fs = fs                       # sample rate
        self.startTime = startTime
        self.minHz = minHz
        self.maxHz = maxHz
        nyquistF = self.fs/2
        fRes = 0.5
        self.nFFT = max(2048, (60*2*nyquistF) / fRes)

    def spectrogram(self, winsize=5):
        """
        Compute the BVP signal spectrogram restricted to the
        band 42-240 BPM by using winsize (in sec) samples.
        """

        # -- spect. Z is 3-dim: Z[#chnls, #freqs, #times]
        F, T, Z = stft(self.data,
                       self.fs,
                       nperseg=self.fs*winsize,
                       noverlap=self.fs*(winsize-self.step),
                       boundary='even',
                       nfft=self.nFFT)
        Z = np.squeeze(Z, axis=0)

        # -- freq subband (0.65 Hz - 4.0 Hz)
        minHz = 0.65
        maxHz = 4.0
        band = np.argwhere((F > minHz) & (F < maxHz)).flatten()
        self.spect = np.abs(Z[band, :])     # spectrum magnitude
        self.freqs = 60*F[band]            # spectrum freq in bpm
        self.times = T                     # spectrum times

        # -- BPM estimate by spectrum
        self.bpm = self.freqs[np.argmax(self.spect, axis=0)]

    def getBPM(self, winsize=5):
        """
        Get the BPM signal extracted from the ground truth BVP signal.
        """
        self.spectrogram(winsize)
        return self.bpm, self.times


def getBPM_by_FFT(signalDict):
    bpmPoints, timeIdxPoints = BVPsignal(signalDict['PPG'], signalDict['SampleRate']).getBPM()
    return timeIdxPoints, bpmPoints

def find_nearest(arr, val):
    return np.abs(arr - val).argmin()


def PPG_Analysis_UBFC_rPPG(subjectDict, debugDirPath=None):
    for subjectKey, signalDict in subjectDict.items():

        # Signal Plot (ax1)
        time_axis = np.linspace(0, signalDict['Duration'], len(signalDict['PPG']))
        peaksIdx = getPeaks(signalDict)
        peaksIdx_fit = fitPeaks_v1(signalDict, peaksIdx)

        # Power Spectrum Plot (ax2)
        freqIdxPoints, powerSpectrumPoints = getPowerSpectrum(signalDict)
        idx = find_nearest(freqIdxPoints, 10.)
        freqIdxPoints = freqIdxPoints[1:idx]
        powerSpectrumPoints = powerSpectrumPoints[1:idx]

        # Heart Rate Plot (ax3)
        gt_timeIdxPoints, gt_bpmPoints = getBPM_from_GT(signalDict)
        peak_timeIdxPoints, peak_bpmPoints, peak_outlierPoints, peak_outlierValues = getBPM_by_Peaks(signalDict)
        fft_timeIdxPoints, fft_bpmPoints = getBPM_by_FFT(signalDict)

        if debugDirPath is not None:
            if not os.path.isdir(debugDirPath):
                print(f'Not Found debugDirPath: {debugDirPath}')
                return

            parameters = {'axes.titlesize': 18}
            plt.rcParams.update(parameters)
            plt.figure(figsize=(20, 8))


            ax1 = plt.subplot(211)
            ax1.set_title(f'UBFC-rPPG PPG signal {subjectKey}')
            ax1.set_xlabel('Time [sec]')
            ax1.margins(0.01)
            ax1.plot(time_axis, signalDict['PPG'])
            # ax1.plot(time_axis[peaksIdx], signalDict['PPG'][peaksIdx], 'ro')
            ax1.plot(time_axis[peaksIdx_fit], signalDict['PPG'][peaksIdx_fit], 'ro')


            ax2 = plt.subplot(223)
            ax2.set_title('Power Spectrum')
            ax2.set_xlabel('Frequency [Hz]')
            ax2.plot(freqIdxPoints, powerSpectrumPoints)


            ax3 = plt.subplot(224)
            ax3.set_title('Heart Rate')
            ax3.set_xlabel('Time [sec]')
            ax3.set_ylim((0, max(max(peak_bpmPoints), max(gt_bpmPoints))+10))
            ax3.plot(gt_timeIdxPoints, gt_bpmPoints, 'b-', linewidth=3.5,
                     label=f'Ground Truth   Mean: {round(gt_bpmPoints[gt_bpmPoints>40].mean(),1)},   Std: {round(gt_bpmPoints[gt_bpmPoints>40].std(),1)}')
            ax3.plot(peak_timeIdxPoints, peak_bpmPoints, 'ro-',
                     label=f'From_Peaks     Mean: {round(peak_bpmPoints.mean(),1)},   Std: {round(peak_bpmPoints.std(),1)}')
            ax3.plot(fft_timeIdxPoints, fft_bpmPoints, 'ko-',
                     label=f'From_FFT       Mean: {round(fft_bpmPoints.mean(), 1)},   Std: {round(fft_bpmPoints.std(), 1)}')
            ax3.plot(peak_outlierPoints, peak_outlierValues, 'rx', label=f'Outlier(Peaks)    {len(peak_outlierPoints)}points')
            ax3.legend(loc='lower left', fontsize=10)

            plt.tight_layout()
            plt.savefig(debugDirPath + f'/{subjectKey}.png')
            # plt.show()
            plt.close()




if __name__ == "__main__":
    rootPath = '/home/jh/data/UBFC'
    debugDirPath = '/home/jh/PycharmProjects/temp/debug_dir/'
    subjectDict = getSubjectDict_UBFC_rPPG(rootPath, subjectList=None)
    PPG_Analysis_UBFC_rPPG(subjectDict, debugDirPath=debugDirPath)








