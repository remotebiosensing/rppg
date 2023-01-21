import numpy as np
import matplotlib.pyplot as plt
'''should be done right after sig_slicing()'''
from scipy.interpolate import make_interp_spline, BSpline, interp1d
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import MinMaxScaler

'''
np.shape(input_sig) : ndarray(702, 7500)     ex) ple, abp
'''

# TODO np.diff 사용 고려

# def Taylor_series(sig, degree=1):
#

class LowPassFilter(object):
    def __init__(self, cut_off_freqency, ts):
        # cut_off_freqency: 차단 주파수
        # ts: 주기

        self.ts = ts
        self.cut_off_freqency = cut_off_freqency
        self.tau = self.get_tau()

        self.prev_data = 0.

    def get_tau(self):
        return 1 / (2 * np.pi * self.cut_off_freqency)

    def filter(self, data):
        val = (self.ts * data + self.tau * self.prev_data) / (self.tau + self.ts)
        self.prev_data = val
        return val

def signal_smoothing(input_sig, k=3):
    x_new = np.linspace(0, len(input_sig), len(input_sig) * 5)
    spl = make_interp_spline(np.arange(input_sig.shape[0]), input_sig, k=k)
    smoothed_signal = spl(x_new)
    return smoothed_signal



def get_derivative(input_sig):
    velocity = np.append(input_sig[1:], input_sig[-1]) - input_sig
    smoothed_vel = gaussian_filter1d(velocity, sigma=3)
    acceleration = np.append(smoothed_vel[1:], smoothed_vel[-1]) - smoothed_vel
    smoothed_acc = gaussian_filter1d(acceleration, sigma=3)
    # smoothed_vel4 = gaussian_filter1d(smoothed_vel, sigma=4)
    # acc4 = np.append(smoothed_vel4[1:], smoothed_vel4[-1]) - smoothed_vel4
    # smoothed_acc4 = gaussian_filter1d(acc4, sigma=4)
    # acceleration = np.append(velocity[1:], velocity[-1]) - velocity
    # f_cubic = interp1d(np.arange(input_sig.shape[0]), velocity, kind='cubic')
    # y_smoothed0 = gaussian_filter1d(velocity, sigma=1)
    # y_smoothed1 = gaussian_filter1d(velocity, sigma=1.5)
    # y_smoothed2 = gaussian_filter1d(velocity, sigma=2)
    # y_smoothed3 = gaussian_filter1d(velocity, sigma=2.5)
    # y_smoothed4 = gaussian_filter1d(velocity, sigma=3)
    # y_smoothed5 = gaussian_filter1d(velocity, sigma=3.5)
    # y_smoothed6 = gaussian_filter1d(velocity, sigma=4)
    # acc_smooted6 = gaussian_filter1d(acceleration, sigma=4)
    # for i in range(1, 7):
    #     plt.plot(eval(f'y_smoothed{i}'))
    # plt.show()
    # ple_temp[-1] = np.mean(ple_temp[-3:-2])
    # vpg = np.append(np.diff(input_sig, axis=0), input_sig[-1] - input_sig[-2])
    # apg = np.append(np.diff(vpg, axis=0), vpg[-1] - vpg[-2])
    # lpf = LowPassFilter(3.0, 1.0)
    # l_velocity = lpf.filter(velocity)
    # l_acceleration = lpf.filter(acceleration)
    # cubic_velocity = f_cubic(np.arange(input_sig.shape[0]))
    # s_velocity = signal_smoothing(velocity)
    return smoothed_vel, smoothed_acc

def channel_cat(input_sig, scale=True):
    if scale:
        min_max_scaler = MinMaxScaler(feature_range=(1, 3))
        input_sig = np.squeeze(min_max_scaler.fit_transform(np.reshape(input_sig, (750, 1))))
    vel, acc = get_derivative(input_sig)
    input_sig = np.expand_dims(input_sig[15:735:2], axis=0)
    vel = np.expand_dims(vel[15:735:2], axis=0)
    acc = np.expand_dims(acc[15:735:2], axis=0)
    cat = np.concatenate((input_sig, vel, acc), axis=0)
    return cat
def diff_np(input_sig, input_sig2=None):
    ple_diff = []
    abp_diff = []
    if input_sig2 is None:
        for p in input_sig:
            ple_temp = np.append(p[1:], p[-1]) - p
            ple_temp[-1] = np.mean(ple_temp[-3:-2])
            ple_diff.append(ple_temp)

        ple_diff = np.array(ple_diff)
        return ple_diff
    else:
        for p, s in zip(input_sig, input_sig2):
            ple_temp = np.append(p[1:], p[-1]) - p
            abp_temp = np.append(s[1:], s[-1]) - s
            ple_temp[-1] = np.mean(ple_temp[-3:-2])
            abp_temp[-1] = np.mean(abp_temp[-3:-2])
            ple_diff.append(ple_temp)
            abp_diff.append(abp_temp)

        ple_diff = np.array(ple_diff)
        abp_diff = np.array(abp_diff)
        return ple_diff, abp_diff


# TODO -> data_aggregator에서 degree에 따른 결과가 가지 수가 안맞음
def diff_channels_aggregator(zero, first=None, second=None):
    zero = np.expand_dims(zero, axis=1)

    if first is None:
        # print('zero called')
        print('channel aggregated ( f ) :', np.shape(zero))
        # print(zero[0])
        return zero

    elif (first is not None) and (second is None):
        # print('first called')
        first = np.expand_dims(first, axis=1)
        temp1 = np.concatenate((zero, first), axis=1)

        print('channel aggregated ( f + f\' ) :', np.shape(temp1))
        # print(temp1[0])
        return temp1
    elif (first is not None) and (second is not None):
        # print('second called')
        first = np.expand_dims(first, axis=1)
        second = np.expand_dims(second, axis=1)
        temp2 = np.concatenate((zero, first, second), axis=1)

        print('channel aggregated ( f + f\' + f\'\' ) :', np.shape(temp2))
        # print(temp2[0])
        # print(temp2)
        return temp2
