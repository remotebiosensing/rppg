import tensorflow as tf
import numpy as np
from scipy import signal
# from rppg.utils.funcs import detrend (이 부분은 필요한 처리에 맞게 수정해야 합니다)
import os
from scipy.signal import firwin, lfilter
from scipy import signal
import cv2
import face_recognition


# 필터 계수 계산
global b, a
b, a = signal.butter(1, [0.75 / 30 * 2, 3 / 30 * 2], btype='bandpass')
print(b, a)

# 필터 계수 계산
global d, c
d, c = signal.butter(1, [0.18 / 30 * 2, 0.5 / 30 * 2], btype='bandpass')
def create_firwin_bandpass(lowcut, highcut, fs, numtaps):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return firwin(numtaps, [low, high], pass_zero=False)

coefficients = create_firwin_bandpass(0.5, 1.5, 30, 64)




# 환경 변수 설정
class ori(tf.Module):
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

    @tf.function(input_signature=[tf.TensorSpec(shape=[256, 3], dtype=tf.float32)])
    def __call__(self, x):
        x = tf.transpose(x, perm=[1, 0])  # (B,C,N,H, W)  (B, N, C)
        x = self.rectangular_smoothing(x[1], 5)
        x = self.detrend_tf(x, 100)  # (B,C,N,H, W)  (B, N, C)
        x = tf.signal.rfft(x, fft_length=[256])
        x = tf.abs(x)
        f_ppg = tf.linspace(0.0, 15, 65)

        # Mask for filtering frequencies
        fmask_ppg = tf.where((f_ppg >= 0.75) & (f_ppg <= 3.5))
        mask_ppg = tf.gather(f_ppg, fmask_ppg)
        mask_pxx = tf.gather(tf.squeeze(x), fmask_ppg)

        # Calculate heart rate
        # print(mask_pxx.shape)
        hr_index = tf.argmax(mask_pxx) + 3
        hr = tf.gather(mask_ppg, hr_index)[0]
        return hr*60

# class POS(tf.Module):
#     def __init__(self):
#         super(POS, self).__init__()
#         self.fs = 30
#         self.WinSec = 1.6
#         self.m_option = tf.constant([[0, 1, -1], [-2, 1, 1]], dtype=tf.float32)
#
#     def set_diag(self, matrix, diag_values):
#         matrix_shape = tf.shape(matrix)
#         diag_len = tf.minimum(matrix_shape[0], matrix_shape[1])
#         diag_indices = tf.stack([tf.range(diag_len), tf.range(diag_len)], axis=1)
#         return tf.tensor_scatter_nd_update(matrix, diag_indices, diag_values)
#
#     def detrend_tf(self, signal, Lambda):
#         signal_length = len(signal)
#
#         # Observation matrix
#         H = tf.eye(signal_length)
#
#         # Second-order difference matrix
#         ones = tf.ones(signal_length - 2)
#         minus_twos = -2 * ones
#         D = tf.zeros([signal_length - 2, signal_length], dtype=tf.float32)
#         D = self.set_diag(D, minus_twos)  # 주 대각선
#         D = self.set_diag(tf.roll(D, shift=1, axis=1), ones)  # 위 대각선
#         D = self.set_diag(tf.roll(D, shift=-1, axis=1), ones)  # 아래 대각선
#
#         # Applying the detrending filter
#         # Replace Matrix Inverse with an approximate method if possible
#         # This is a placeholder for the inverse operation
#         # You might need to implement an approximate method
#         H_inv = tf.linalg.inv(H + Lambda ** 2 * tf.matmul(D, D, transpose_a=True))
#         filtered_signal = tf.matmul(H - H_inv, tf.reshape(signal, [-1, 1]))
#
#         return tf.reshape(filtered_signal, [-1])
#
#     def apply_butterworth_filter(self, signal, b, a):
#         # b 계수로 필터 커널 생성
#         kernel = tf.constant(b[::-1], dtype=tf.float32, shape=[len(b), 1, 1])
#
#         # a 계수에 따라 신호 조정
#         filtered_signal = signal
#         for i in range(1, len(a)):
#             filtered_signal = filtered_signal - a[i] * tf.concat([tf.zeros([i], dtype=tf.float32), signal[:-i]], axis=0)
#
#         # 필터 적용
#         filtered_signal = tf.nn.conv1d(tf.reshape(filtered_signal, [1, -1, 1]), kernel, stride=1, padding='VALID')
#         return tf.squeeze(filtered_signal)
#
#     def _nearest_power_of_2(self, x):
#         return 2 ** tf.cast(tf.math.round(tf.math.log(tf.cast(x, tf.float32)) / tf.math.log(2.0)), tf.int32)
#
#     def calculate_hr_tf(self, ppg_signal, fs=60.0, low_pass=0.75, high_pass=2.5):
#         ppg_signal = tf.expand_dims(ppg_signal, 0)
#         N = self._nearest_power_of_2(tf.shape(ppg_signal)[1])
#         fft_ppg = tf.signal.rfft(ppg_signal, fft_length=[N])
#         pxx_ppg = tf.abs(fft_ppg) ** 2
#
#         # Generate frequency array
#         f_ppg = tf.linspace(0.0, fs / 2, N // 2 + 1)
#
#         # Mask for filtering frequencies
#         fmask_ppg = tf.where((f_ppg >= low_pass) & (f_ppg <= high_pass))
#         mask_ppg = tf.gather(f_ppg, fmask_ppg)
#         mask_pxx = tf.gather(tf.squeeze(pxx_ppg), fmask_ppg)
#
#         # Calculate heart rate
#         hr_index = tf.argmax(mask_pxx, axis=0)
#         hr = tf.gather(mask_ppg, hr_index)[0] * 60
#
#         return hr
#
#
#     def apply_fir_filter(self,signal, coefficients):
#         # 1D FIR 필터 적용을 위해 신호를 2D로 확장합니다 (batch_size, signal_length).
#         #signal_tf = tf.constant(signal, dtype=tf.float32)
#         #coefficients_tf = tf.constant(coefficients, dtype=tf.float32)
#         coefficients = tf.convert_to_tensor(coefficients, dtype=tf.float32)
#         # 필터 계수를 적용합니다.
#         filtered_signal = tf.nn.conv1d(signal[tf.newaxis, :, tf.newaxis],
#                                        coefficients[:, tf.newaxis, tf.newaxis],
#                                        stride=1, padding='SAME')
#         return filtered_signal[0, :, 0]
#
#
#     def extract_envelope(self, signal, frame_length):
#         # 신호를 2차원으로 확장합니다 (batch_size, length).
#         signal = tf.expand_dims(signal, axis=0)
#         # Ensure signal has a channel dimension [batch_size, length, channels]
#         signal_2d = tf.expand_dims(signal, axis=-1)
#
#         # Use max_pool1d with the correct input shape
#         max_pooled_signal = tf.nn.max_pool1d(input=signal_2d, ksize=frame_length, strides=1, padding='SAME')
#         # Assuming we also want to perform min pooling to get the lower envelope
#         min_pooled_signal = -tf.nn.max_pool1d(input=-signal_2d, ksize=frame_length, strides=1, padding='SAME')
#
#         # Remove the extra dimensions to get back to 1D signal
#         upper_envelope = tf.squeeze(max_pooled_signal, axis=[0, -1])
#         lower_envelope = tf.squeeze(min_pooled_signal, axis=[0, -1])
#
#         return upper_envelope, lower_envelope
#
#     def calculate_frequency_components(self, signal, fs):
#         # FFT를 수행합니다.
#         signal, _ = self.pad_to_power_of_two(signal)
#         fft = tf.signal.rfft(signal)
#         signal_length = tf.size(signal)
#         fft_length = tf.cast(signal_length / 2 + 1, tf.int32)
#
#         # 주파수 배열을 수동으로 계산합니다.
#         fft_freqs = tf.linspace(0.0, fs / 2, fft_length)
#         fft_amplitudes = tf.abs(fft)
#
#         # 2Hz 기준으로 낮은 쪽(LF)과 높은 쪽(HF)의 진폭을 계산합니다.
#         lf_amplitudes = tf.boolean_mask(fft_amplitudes, fft_freqs < 2)
#         hf_amplitudes = tf.boolean_mask(fft_amplitudes, fft_freqs >= 2)
#         lf = tf.reduce_sum(lf_amplitudes)
#         hf = tf.reduce_sum(hf_amplitudes)
#
#         return lf, hf
#     def calculate_hf_lf_ratio(self,signal, fs):
#         lf, hf = self.calculate_frequency_components(signal, fs)
#         # 0으로 나누기를 방지하기 위한 작은 값을 추가합니다.
#         hf_lf_ratio = hf / (lf + 1e-10)
#         return hf_lf_ratio
#
#     def process_signals(self, detrended_R, detrended_B, R_envelope_mean, B_envelope_mean):
#         # 'out_R'에 대한 조건에 따라 R_envelope_mean 값으로 구성된 텐서 생성
#         r_condition = tf.abs(detrended_R) < R_envelope_mean
#         r_values = tf.where(r_condition, R_envelope_mean, tf.zeros_like(detrended_R))
#
#         # 'out'에 대한 조건에 따라 B_envelope_mean 값으로 구성된 텐서 생성
#         b_condition = tf.abs(detrended_B) < B_envelope_mean
#         b_values = tf.where(b_condition, B_envelope_mean, tf.zeros_like(detrended_B))
#
#         return r_values, b_values
#
#     def detect_peaks(self,signal):
#         # 피크 검출을 위해 각 요소가 이전 및 다음 요소보다 큰지 비교합니다.
#         half_window = 3 // 2
#
#         # 피크를 저장할 텐서를 초기화합니다.
#         padded_signal = tf.pad(signal, [[half_window, half_window]], "CONSTANT")
#
#         # 윈도우 내에서 최대값이 현재 값과 같은지 여부를 판별하는 함수
#         def is_peak(i):
#             window = padded_signal[i:i + 3]
#             return tf.cast(tf.equal(tf.reduce_max(window), signal[i - half_window]), tf.int32)
#
#         # 각 위치에서 피크 여부를 판별합니다.
#         peaks = tf.map_fn(is_peak, tf.range(half_window, len(signal) + half_window), dtype=tf.int32)
#
#         # 피크 위치의 인덱스를 반환합니다.
#         return tf.where(tf.equal(peaks, 1))[:, 0]
#
#     def calculate_through_dft(self, signal, peaks):
#         # 첫 번째 값으로 나누어 정규화합니다.
#         normalized_signal = signal / signal[0]
#
#         # 피크 위치의 값을 추출합니다.
#         peak_values = tf.gather(normalized_signal, peaks)
#
#         return tf.abs(peak_values)
#
#     def perform_fft(self,signal):
#         # 신호에 대한 FFT 수행
#
#         signal, _ = self.pad_to_power_of_two(signal)
#
#         fft_complex = tf.signal.rfft(signal)
#
#         # FFT의 절대값 계산 (진폭)
#         fft_abs = tf.abs(fft_complex)
#
#         return fft_abs
#
#     def pad_to_power_of_two(self,signal):
#         # 2의 거듭제곱 중에서 가장 가까운 값 찾기
#         next_power_of_two = 2 ** tf.math.ceil(tf.math.log(tf.cast(tf.size(signal), tf.float32)) / tf.math.log(2.0))
#         next_power_of_two = tf.cast(next_power_of_two, tf.int32)
#
#         # 필요한 만큼 패딩 추가
#         pad_size = next_power_of_two - tf.size(signal)
#         signal_padded = tf.pad(signal, [[0, pad_size]], 'CONSTANT')
#
#         return signal_padded, next_power_of_two
#
#     @tf.function(input_signature=[tf.TensorSpec(shape=[1, 3, 512, 50, 50], dtype=tf.float32)])
#     def __call__(self, x):
#         x = tf.transpose(x, perm=[0, 2, 1, 3, 4])  # (B, N, C)
#         x = tf.reduce_mean(x, axis=[3, 4])
#
#         batch_size, N, num_features = x.shape
#         H = tf.zeros((batch_size, 1, N))
#
#         #SPo2
#
#         filtered_R = self.apply_fir_filter(x[0][0],coefficients)
#         filtered_B = self.apply_fir_filter(x[0][2],coefficients)
#
#         up_r, _ = self.extract_envelope(filtered_R,256)
#         up_b, _ = self.extract_envelope(filtered_B, 256)
#
#         R_envelope_mean = tf.reduce_mean(up_r)
#         B_encelope_mean = tf.reduce_mean(up_b)
#
#         det_r = self.detrend_tf(filtered_R,100)
#         det_b = self.detrend_tf(filtered_B,100)
#
#         r_v, b_v = self.process_signals(det_r,det_b,R_envelope_mean,B_encelope_mean)
#
#         out_dft_r_real = self.perform_fft(r_v)
#         out_dft_b_real = self.perform_fft(b_v)
#
#         peaks_r = self.detect_peaks(out_dft_r_real)
#         through_dft_r = self.calculate_through_dft(out_dft_r_real, peaks_r)
#
#         # B 신호에 대한 피크 검출
#         peaks_b = self.detect_peaks(out_dft_b_real)
#         through_dft_b = self.calculate_through_dft(out_dft_b_real, peaks_b)
#
#         spo2_len = tf.minimum(tf.size(through_dft_r), tf.size(through_dft_b))
#
#         # 두 텐서를 spo2_len 길이로 자릅니다.
#         through_dft_r = through_dft_r[:spo2_len]
#         through_dft_b = through_dft_b[:spo2_len]
#
#         # SpO2 계산
#         spo2_raw = 96.58 - -0.015 * through_dft_r / through_dft_b * 100
#
#         # 조건에 따라 SpO2 값 필터링
#         spo2_filtered = tf.boolean_mask(spo2_raw, spo2_raw < 100)
#
#         # 평균 SpO2 값 계산
#         spo2_avg = tf.reduce_mean(spo2_filtered)
#
#
#
#         for b in tf.range(batch_size):
#             RGB = x[b]  # Assume RGB preprocessing already done
#             N = tf.shape(RGB)[0]
#             l = int(self.fs * self.WinSec)  # math.ceil(WinSec * fs)
#
#             for n in range(N):
#                 m = n - l
#                 if m >= 0:
#                     Cn = RGB[m:n, :] / tf.reduce_mean(RGB[m:n, :], axis=0)
#                     Cn = tf.transpose(Cn)
#                     S = tf.matmul(self.m_option, Cn)
#                     h = S[0, :] + (tf.math.reduce_std(S[0, :]) / tf.math.reduce_std(S[1, :])) * S[1, :]
#                     mean_h = tf.reduce_mean(h)
#                     h = h - mean_h
#                     indices = tf.reshape(tf.range(m, n), [-1, 1])
#                     indices = tf.concat([tf.fill([n - m, 1], b), tf.zeros([n - m, 1], dtype=tf.int32), indices], axis=1)
#                     H = tf.tensor_scatter_nd_add(H, indices, tf.reshape(h, [-1]))
#         # #
#         BVP = tf.squeeze(H)
#         hr = []
#         rr = []
#         hf_lf_ratio = []
#         BVP = self.detrend_tf(BVP, 100)
#         #for i in range(len(BVP)):
#         # BVP = tf.tensor_scatter_nd_update(
#         #     BVP,
#         #     [0],  # 인덱스를 나타내는 텐서
#         #     [self.detrend_tf(BVP, 100)]  # 업데이트할 값
#         # )
#         hr.append(self.calculate_hr_tf(BVP, 30,0.8, 3))
#         rr.append(self.calculate_hr_tf(BVP, 30, 0.18, 0.5))
#         hf_lf_ratio.append(self.calculate_hf_lf_ratio(BVP, 30))
#
#         # # 아래 부분은 필요한 신호 처리에 따라 수정해야 합니다.
#         # # BVP 처리 코드...
#         # hr과 rr의 평균 값을 계산
#         hr_mean = tf.reduce_mean(tf.stack(hr))
#         rr_mean = tf.reduce_mean(tf.stack(rr))
#         hf_lf_ratio_mean = tf.reduce_mean(tf.stack(hf_lf_ratio))
#
#         # 결과를 텐서로 변환하고 [1, 2] 형태로 리쉐이핑
#         result = tf.stack([hr_mean, rr_mean,hf_lf_ratio_mean,spo2_avg])
#         result = tf.reshape(result, [1, 4])
#
#         # BVP = tf.reshape(BVP, (batch_size, -1))
#         return result#, spo2_avg#up_r#result#up_r#result


if __name__ == "__main__":




    model = ori()
    # model.__call__(input_data)?
    concrete_func = model.__call__.get_concrete_function()
    # tf.summary.trace_on(graph=True, profiler=True)

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], model)
    # TF Select를 사용하여 TensorFlow 연산을 포함시킵니다.
    tflite_model = converter.convert()

    # Save the model.
    with open('model2.tflite', 'wb') as f:
        f.write(tflite_model)

        interpreter = tf.lite.Interpreter(model_path='model2.tflite')
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)


