import tensorflow as tf
import numpy as np
from scipy import signal
# from rppg.utils.funcs import detrend (이 부분은 필요한 처리에 맞게 수정해야 합니다)
import os
from scipy.signal import firwin, lfilter
from scipy import signal
import cv2
import face_recognition
from matplotlib import pyplot as plt
from scipy.sparse import spdiags

frame_num = 512

def detrend(signal, Lambda):
    """detrend(signal, Lambda) -> filtered_signal
    This function applies a detrending filter.
    This  is based on the following article "An advanced detrending method with application
    to HRV analysis". Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.
    *Parameters*
      ``signal`` (1d numpy array):
        The signal where you want to remove the trend.
      ``Lambda`` (int):
        The smoothing parameter.
    *Returns*
      ``filtered_signal`` (1d numpy array):
        The detrended signal.
    """
    signal_length = len(signal)

    # observation matrix
    H = np.identity(signal_length)

    # second-order difference matrix

    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot((H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), signal)
    return filtered_signal


def tflite_converting(model, file_name):
    concrete_func = model.__call__.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], model)
    tflite_model = converter.convert()
    with open(file_name, 'wb') as f:
        f.write(tflite_model)


def test_tflite(input_data, file_name):
    interpreter = tf.lite.Interpreter(model_path=file_name)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(input_details)
    print(output_details)

    # input_data = np.array(np.random.random_sample([1, 3, 512, 50, 50]), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)
    print(np.shape(output_data))
    return output_data


class Transpose_test(tf.Module):
    def __init__(self):
        super().__init__()

    @tf.function(input_signature=[tf.TensorSpec(shape=[512, 3], dtype=tf.float32)])
    def __call__(self, x):
        x = tf.transpose(x, perm=[1, 0])  # (B,C,N,H, W)  (B, N, C)
        return x


class Smoothing_test(tf.Module):
    def __init__(self):
        super().__init__()

    def rectangular_smoothing(self,signal, window_size, mode='valid'):
        # Rectangular kernel 생성 (모든 값이 1/window_size)
        kernel = tf.ones(shape=[window_size], dtype=tf.float32) / window_size

        # 신호를 2D로 확장 (1D 컨볼루션을 위함)
        signal_2d = tf.reshape(signal, [1, -1, 1])

        # 커널을 3D로 확장
        kernel_3d = tf.reshape(kernel, [window_size, 1, 1])

        # 1D 컨볼루션을 사용하여 smoothing 수행
        smoothed_signal = tf.nn.conv1d(signal_2d, kernel_3d, stride=1, padding=mode.upper())

        # 결과를 1D로 재구성
        return tf.reshape(smoothed_signal, [-1])

    @tf.function(input_signature=[tf.TensorSpec(shape=[3, 512], dtype=tf.float32)])
    def __call__(self, x):
        x = self.rectangular_smoothing(x[1],5)
        return x

class Detrend_test(tf.Module):
    def __init__(self):
        super().__init__()

    def set_diag(self, matrix, diag_values):
        matrix_shape = tf.shape(matrix)
        diag_len = tf.minimum(matrix_shape[0], matrix_shape[1])
        diag_indices = tf.stack([tf.range(diag_len), tf.range(diag_len)], axis=1)
        return tf.tensor_scatter_nd_update(matrix, diag_indices, diag_values)

    def detrend_tf(self, signal, Lambda):
        signal_length = len(signal)
        H = tf.eye(signal_length)

        # second-order difference matrix
        diagonals = [tf.ones(signal_length), -2 * tf.ones(signal_length), tf.ones(signal_length)]

        D = tf.linalg.diag(diagonals[0], k=0) + tf.linalg.diag(diagonals[1], k=1)[:-1, :-1] + tf.linalg.diag(diagonals[2],k=2)[:-2, :-2]
        D = D[:-2, :]  # Adjust shape to (signal_length - 2, signal_length)

        filtered_signal = tf.matmul((H - tf.linalg.inv(H + Lambda ** 2 * tf.matmul(D, D, transpose_a=True))),
                                    signal[:, tf.newaxis])

        return tf.squeeze(filtered_signal)


    @tf.function(input_signature=[tf.TensorSpec(shape=[508], dtype=tf.float32)])
    def __call__(self, x):
        x = self.detrend_tf(x,100)  # (B,C,N,H, W)  (B, N, C)
        return x

class Fft_test(tf.Module):
    def __init__(self):
        super().__init__()

    def _nearest_power_of_2(self, x):
        return 2 ** tf.cast(tf.math.round(tf.math.log(tf.cast(x, tf.float32)) / tf.math.log(2.0)), tf.int32)

    @tf.function(input_signature=[tf.TensorSpec(shape=[508], dtype=tf.float32)])
    def __call__(self, x):

        x = tf.signal.rfft(x, fft_length=[512])
        x = tf.abs(x)
        return x

class Calculate_HR(tf.Module):
    def __init__(self):
        super().__init__()

    @tf.function(input_signature=[tf.TensorSpec(shape=[257], dtype=tf.float32)])
    def __call__(self, signal, fs=30, low_pass=0.75, high_pass=2.5 ):
        f_ppg = tf.linspace(0.0, frameRate//2, 257)

        # Mask for filtering frequencies
        fmask_ppg = tf.where((f_ppg >= low_pass) & (f_ppg <= high_pass))
        mask_ppg = tf.gather(f_ppg, fmask_ppg)
        mask_pxx = tf.gather(tf.squeeze(signal), fmask_ppg)

        # Calculate heart rate
        # print(mask_pxx.shape)
        hr_index = tf.argmax(mask_pxx) + 7
        hr = tf.gather(mask_ppg, hr_index)[0]

        # f_ppg1 = tf.linspace(0.0, 15, 65)
        # fmask_ppg1 = tf.where((f_ppg1 >= 0.8) & (f_ppg1 <= 3.5))
        # mask_ppg1 = tf.gather(f_ppg1, fmask_ppg1)
        # mask_pxx1 = tf.gather(tf.squeeze(out_fft), fmask_ppg1)
        # hr_index = tf.argmax(mask_pxx1)
        # hr1 = tf.gather(mask_ppg1, hr_index)[0]

        return hr

if __name__ == "__main__":


    if os.path.isfile("C:\data\\mobile\\5.mp4"):
        cap = cv2.VideoCapture("C:\data\\mobile\\5.mp4")

    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 영상의 넓이(가로) 프레임
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 영상의 높이(세로) 프레임

    frame_size = (frameWidth, frameHeight)

    frameRate = 30

    data = []

    cnt = 0

    skip = 0

    while True:
        retval, frame = cap.read()
        if not (retval):  # 프레임정보를 정상적으로 읽지 못하면
            break  # while문을 빠져나가기

        face_locations = face_recognition.face_locations(frame, 1, model='hog')
        if len(face_locations) >= 1:
            (bottom, right, top, left) = face_locations[0]
            tmp = cv2.resize(frame[bottom:top, left:right], (1, 1))
            data.append(tmp)
            cnt += 1
        else:
            if len(data) > 0:
                data.append(data[-1])
                cnt += 1
            else:
                continue
            #exit(0)

        if cnt <= skip:
            continue

        print(cnt)
        if cnt == frame_num+skip:
            break

        # cv2.imshow('frame', frame[bottom:top,left:right])  # 프레임 보여주기
        # key = cv2.waitKey(frameRate)  # frameRate msec동안 한 프레임을 보여준다

        # 키 입력을 받으면 키값을 key로 저장 -> esc == 27(아스키코드)
        # if key == 27:
        #    break  # while문을 빠져나가기

    if cap.isOpened():  # 영상 파일(카메라)이 정상적으로 열렸는지(초기화되었는지) 여부
        cap.release()  # 영상 파일(카메라) 사용을 종료

    cv2.destroyAllWindows()

    # model = POS()
    input_data = tf.squeeze(tf.convert_to_tensor(
        data))  # tf.transpose(tf.reshape(tf.convert_to_tensor(data),(1,48,50,50,3)),(0,4,1,2,3))#np.array(np.random.random_sample([1, 3, 512, 50, 50]), dtype=np.float32)
    input_data = tf.cast(input_data, tf.float32)
    mean = tf.reduce_mean(input_data, axis=0)
    stddev = tf.math.reduce_std(input_data, axis=0)
    input_data = (input_data - mean) / stddev

    # Calculate the padding sizes
    # padding_size = 100 - input_data.shape[0]

    # Create the padding configuration
    # The format is [(before_1, after_1), (before_2, after_2), ...]
    # For your case, you want to add padding before and after the first dimension
    # padding = [[padding_size, 0], [0, 0]]
    # padded_tensor = tf.pad(input_data, padding)

    # input_data = tf.cast(padded_tensor,tf.float32)
    print("A")

    transpose_test = Transpose_test()
    tflite_converting(transpose_test, "transpose.tflite")
    out_transpose = test_tflite(input_data, "transpose.tflite")
    plt.plot(out_transpose[1])
    plt.show()

    smoothing_test = Smoothing_test()
    tflite_converting(smoothing_test, "smoothing.tflite")
    out_smoothing = test_tflite(out_transpose, "smoothing.tflite")

    plt.plot(out_smoothing)
    plt.show()

    detrend_test = Detrend_test()
    tflite_converting(detrend_test, "detrend.tflite")
    out_detrend = test_tflite(out_smoothing, "detrend.tflite")

    plt.plot(out_detrend)
    plt.show()

    fft_test = Fft_test()
    tflite_converting(fft_test, "fft.tflite")
    out_fft = test_tflite(out_detrend, "fft.tflite")

    plt.plot(out_fft)
    plt.show()
    #
    f_ppg1 = tf.linspace(0.0, frameRate//2, 257)
    fmask_ppg1 = tf.where((f_ppg1 >= 0.8) & (f_ppg1 <= 3.5))
    mask_ppg1 = tf.gather(f_ppg1, fmask_ppg1)
    mask_pxx1 = tf.gather(tf.squeeze(out_fft), fmask_ppg1)
    hr_index = tf.argmax(mask_pxx1)
    hr1 = tf.gather(mask_ppg1, hr_index)[0]*60


    ppg_signal = np.expand_dims(out_detrend, 0)
    #N = _nearest_power_of_2(ppg_signal.shape[1])
    f_ppg, pxx_ppg = signal.periodogram(ppg_signal, fs=frameRate//2, nfft=512, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= 0.8) & (f_ppg <= 3.5))
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60


    cal_test = Calculate_HR()
    tflite_converting(cal_test, "cal_test.tflite")
    cal = test_tflite(out_fft, "cal_test.tflite")

    #plt.plot(out_fft)
    #plt.show()

    print(cal*60)
    print(hr)
    print(hr1)
