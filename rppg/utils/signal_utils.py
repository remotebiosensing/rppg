import numpy as np
from scipy.signal import butter, lfilter, get_window

import heartpy.peakdetection as hp_peak
from heartpy.datautils import rolling_mean
from heartpy.filtering import filter_signal


def not_flat_signal_detector(target_signal):
    return np.argwhere(np.diff(target_signal) != 0)


def nan_detector(target_signal):
    nan_idx = np.argwhere(np.isnan(target_signal))
    return nan_idx


def discrete_index_detector(target_index):
    return np.where(np.diff(target_index) != 1)


def not_nan_detector(target_signal):
    not_nan_idx = np.argwhere(~np.isnan(target_signal))
    return not_nan_idx


def signals_slice_by_flat(target_signals):
    # input : signal list [signal1, signal2, ...]
    return_signal_list = []
    valid_flag = False
    for target_signal in target_signals:
        not_flat_index = not_flat_signal_detector(target_signal[0]).reshape(-1)
        discrete_index = discrete_index_detector(not_flat_index)
        if len(discrete_index[0]) == 0:
            continue
        else:
            discrete_index = discrete_index[0].reshape(-1)
        if discrete_index[0] != 0:
            discrete_index = np.insert(discrete_index, 0, -1)
        for i in range(len(discrete_index) - 1):
            valid_flag = True
            return_signal_list.append(
                target_signal[:, not_flat_index[discrete_index[i] + 1]:not_flat_index[discrete_index[i + 1]]])
    if valid_flag:
        return True, return_signal_list
    else:
        return False, None


def not_flat_signal_checker(target_signal, t=2, threshold=0.1, slicing=True):
    # return True if not flat
    if slicing:
        target_signal = signal_slicer(target_signal, t=t, overlap=0)
        for sliced_signal in target_signal:
            if np.std(sliced_signal) < threshold:
                return False
    else:
        if np.std(target_signal) < threshold:
            return False
    return True


def not_nan_checker(target_signal):
    return ~np.isnan(target_signal).any()


def nan_interpolator(target_signal):
    # not used
    nan_idx = nan_detector(target_signal)
    nan_idx = nan_idx.reshape(-1)
    for idx in nan_idx:
        target_signal[idx] = np.nanmean(target_signal)
    return target_signal


def signal_slice_by_nan(target_signal):
    not_nan_index = not_nan_detector(target_signal[0]).reshape(-1)
    discrete_index = discrete_index_detector(not_nan_index)
    return_signal_list = []
    if len(discrete_index[0]) == 0:
        return False, None
    else:
        discrete_index = discrete_index[0].reshape(-1)
    if discrete_index[0] != 0:
        discrete_index = np.insert(discrete_index, 0, -1)
    for i in range(len(discrete_index) - 1):
        return_signal_list.append(
            target_signal[:, not_nan_index[discrete_index[i] + 1]:not_nan_index[discrete_index[i + 1]]])
    return True, return_signal_list


def nan_signal_slicer(target_signal):
    nan_idx = nan_detector(target_signal)
    nan_idx = nan_idx.reshape(-1)
    nan_idx = np.append(nan_idx, len(target_signal))
    nan_idx = np.insert(nan_idx, 0, 0)
    nan_idx = nan_idx.reshape(-1, 2)
    for idx in nan_idx:
        yield target_signal[idx[0]:idx[1]]


"""Nan Detecting Modules"""


def nan_checker(target_signal):
    # return if signal has nan
    return np.isnan(target_signal).any(), len(np.where(np.isnan(target_signal))[0])


"""length checking modules"""


def length_checker(target_signal, length):
    if len(target_signal) < length:
        return False
    else:
        return True


"""peak analysing modules"""


def SBP_DBP_arranger(ABP, SBP, DBP):
    i = 0
    total = len(SBP) - 1
    while i < total:
        flag = False
        # Distinguish SBP[i] < DBP < SBP[i+1]
        for idx_dbp in DBP:
            # Normal situation
            if (SBP[i] < idx_dbp) and (idx_dbp < SBP[i + 1]):
                flag = True
                break
            # abnormal situation
        if flag:
            i += 1
        else:
            # compare peak value
            # delete smaller one @SBP
            if ABP[SBP[i]] < ABP[SBP[i + 1]]:
                SBP = np.delete(SBP, i)
            else:
                SBP = np.delete(SBP, i + 1)
            total -= 1

    i = 0
    total = len(DBP) - 1
    while i < total:
        flag = False
        # Distinguish DBP[i] < SBP < DBP[i+1]
        for idx_sbp in SBP:
            # Normal situation
            if (DBP[i] < idx_sbp) and (idx_sbp < DBP[i + 1]):
                flag = True
                break
        # normal situation
        if flag:
            i += 1
        # abnormal situation, there is no SBP between DBP[i] and DBP[i+1]
        else:
            # compare peak value
            # delete bigger one @DBP
            if ABP[DBP[i]] < ABP[DBP[i + 1]]:
                DBP = np.delete(DBP, i + 1)
            else:
                DBP = np.delete(DBP, i)
            total -= 1

    return SBP, DBP


def peak_detector(target_signal, rol_sec, fs=125.0):
    roll_mean = rolling_mean(target_signal, rol_sec, fs)
    peak_heartpy = hp_peak.detect_peaks(target_signal, roll_mean, ma_perc=20, sample_rate=fs)
    return peak_heartpy['peaklist']


def bottom_detector(target_signal, rol_sec, fs=125.0):
    target_signal = -target_signal
    roll_mean = rolling_mean(target_signal, rol_sec, fs)
    peak_heartpy = hp_peak.detect_peaks(target_signal, roll_mean, ma_perc=20, sample_rate=fs)
    return peak_heartpy['peaklist']


def correlation_checker(target_signal, reference_signal):
    return np.corrcoef(target_signal, reference_signal)[0, 1]


"""flag signal checking modules"""


def flat_signal_checker(target_signal, t=2, threshold=0.1, slicing=True):
    # return True if flat
    if slicing:
        target_signal = signal_slicer(target_signal, t=t, overlap=0)
        for sliced_signal in target_signal:
            if np.std(sliced_signal) < threshold:
                return True
    else:
        if np.std(target_signal) < threshold:
            return True
    return False


def signal_slicer(target_signal, fs=125, t=8, overlap=2):
    return_signal_list = []
    while length_checker(target_signal, t * fs):
        return_signal_list.append(target_signal[:t * fs])
        target_signal = target_signal[(t - overlap) * fs:]
    return np.array(return_signal_list)


def get_cycle_length(signal, rolling_sec=1.5, fs=30.0):
    """
    Get cycle length of signal
    :param signal: signal
    :return: cycle length of signal
    """
    signal = filter_signal(signal, cutoff=8, sample_rate=fs, order=4, filtertype='lowpass')
    peaks = peak_detector(signal, rolling_sec, fs)
    bottoms = bottom_detector(signal, rolling_sec, fs)
    peaks, bottoms = SBP_DBP_arranger(signal, peaks, bottoms)
    cycle_length = (np.mean(np.diff(peaks)) + np.mean(np.diff(bottoms))) / 2
    return int(cycle_length)


def get_rgb_mean(video):
    """
    Get timewise mean of channels
    :param video: video ( T x H x W x C )
    :return: timewise mean of channels and mean of mean
    """
    return np.mean(video, axis=(1, 2))


def get_correlation(x, y):
    """
    Get correlation of x and y
    :param x: x
    :param y: y
    :return: correlation of x and y
    """
    corr = np.corrcoef(x, y)
    return corr[0][1]


def get_channels_mean(video, num_channel):
    """
    Get timewise mean of channels
    :param video: video ( T x H x W x C )
    :param num_channel: channel number
    :return: timewise mean of channels
    """

    return np.mean(video[:, :, :, 1], axis=(1, 2))


def replace_minus_by_zero(x):
    """
    Replace -n by 0
    :param x: input
    :return: output
    """
    return x * (x > 0)


def chrom(BGR_signal, fps, interval_length=None):
    # Количество кадров в исходных данных
    num_frames = len(BGR_signal)

    # Проверка допустимости размера исходных данных
    if (num_frames == 0):
        # Массив исходных данных пуст
        raise NameError('EmptyData')

    # Проверка допустимости значения fps
    if (fps < 9):
        # Недопустимое значение fps, для работы полосового фильтра требуется fps>=9
        raise NameError('WrongFPS')

    # Проверка допустимости значения ширины окна Хеннинга и установка значения в случае допустимого
    if (interval_length == None):
        # В статье Haan2013 использовались записи в 20 кадров/сек при ширине окна равной 32
        # Для сохранения пропорций введен множитель 32/20 (при fps = 20, получим окно шириной 32)
        interval_size = int(fps * (32.0 / 20.0))
    elif (interval_length > 0):
        interval_size = interval_length // 1
    else:
        # Недопустимое значение ширины окна Хеннинга, значение должно быть не менее 32
        raise NameError('WrongIntervalLength')

    # Проверка допустимости размера исходных данных
    if (num_frames < interval_size):
        # Недопустимая длина исходных данных, длина исходных данных должна быть не меньше окна Хеннинга
        raise NameError('NotEnoughData')

    # Разделение исходного сигнала на каналы R,G,B
    R = BGR_signal[:, 2]
    G = BGR_signal[:, 1]
    B = BGR_signal[:, 0]

    # Функция полосовой фильтрации
    def bandpass_filter(data, lowcut, highcut):
        fs = fps  # Частота дискретизации (количество измерений сигнала в 1 сек)
        nyq = 0.5 * fs  # Частота Найквиста
        low = float(lowcut) / float(nyq)
        high = float(highcut) / float(nyq)
        order = 6.0  # Номер фильтра в scipy.signal.butter
        b, a = butter(order, [low, high], btype='band')
        bandpass = lfilter(b, a, data)
        return bandpass

    # -------------------------------------------------------------------
    # Функция вычисления сигнала S на интервале
    def S_signal_on_interval(low_limit, high_limit):

        # Выделение отрывков R,G,B на интервале и их нормализация
        if (low_limit < 0.0):
            num_minus = abs(low_limit)
            R_interval = np.append(np.zeros(num_minus), R[0:high_limit + 1])
            R_interval_norm = R_interval / R_interval[num_minus:interval_size].mean()
            G_interval = np.append(np.zeros(num_minus), G[0:high_limit + 1])
            G_interval_norm = G_interval / G_interval[num_minus:interval_size].mean()
            B_interval = np.append(np.zeros(num_minus), B[0:high_limit + 1])
            B_interval_norm = B_interval / B_interval[num_minus:interval_size].mean()
        elif (high_limit > num_frames):
            num_plus = high_limit - num_frames
            R_interval = np.append(R[low_limit:num_frames], np.zeros(num_plus + 1))
            R_interval_norm = R_interval / R_interval[0:interval_size - num_plus - 1].mean()
            G_interval = np.append(G[low_limit:num_frames], np.zeros(num_plus + 1))
            G_interval_norm = G_interval / G_interval[0:interval_size - num_plus - 1].mean()
            B_interval = np.append(B[low_limit:num_frames], np.zeros(num_plus + 1))
            B_interval_norm = B_interval / B_interval[0:interval_size - num_plus - 1].mean()
        else:
            R_interval = R[low_limit:high_limit + 1]
            R_interval_norm = R_interval / R_interval.mean()
            G_interval = G[low_limit:high_limit + 1]
            G_interval_norm = G_interval / G_interval.mean()
            B_interval = B[low_limit:high_limit + 1]
            B_interval_norm = B_interval / B_interval.mean()

            # Вычисление составляющих Xs и Ys
        Xs, Ys = np.zeros(interval_size), np.zeros(interval_size)
        Xs = 3.0 * R_interval_norm - 2.0 * G_interval_norm
        Ys = 1.5 * R_interval_norm + G_interval_norm - 1.5 * B_interval_norm

        # Вызов функции фильтрации (фильтрация Xs и Ys полосовым фильтром от 0.5 до 4 Гц)
        Xf = bandpass_filter(Xs, 0.5, 4.0)
        Yf = bandpass_filter(Ys, 0.5, 4.0)

        # Вычисление сигнала S до применения окна Хеннинга
        alpha = Xf.std() / Yf.std()
        S_before = Xf - alpha * Yf

        return S_before

    # -------------------------------------------------------------------

    # Поиск количества интервалов
    number_interval = 2.0 * num_frames / interval_size + 1
    number_interval = int(number_interval // 1)

    # Поиск границ интервалов и вычисление в них итогового сигнала
    intervals = []
    S_before_on_interval = []
    for i in range(int(number_interval)):
        i_low = int((i - 1) * interval_size / 2.0 + 1)
        i_high = int((i + 1) * interval_size / 2.0)
        intervals.append([i_low, i_high])
        S_before_on_interval.append(S_signal_on_interval(i_low, i_high))

        # Вычисление окна Хеннинга
    wh = get_window('hamming', interval_size)

    # Поиск индексов точек, в которых нет пересечения окон Хеннинга
    index_without_henning = []
    # Слева
    for i in range(intervals[0][0], intervals[1][0], 1):
        if (i >= 0):
            index_without_henning.append(i)
    # Справа
    for i in range(intervals[len(intervals) - 2][1] + 1, intervals[len(intervals) - 1][1], 1):
        if (i <= num_frames):
            index_without_henning.append(i)

    # Расчет итогового сигнала
    S_after = np.zeros(num_frames)
    for i in range(num_frames):
        for j in intervals:
            if (i >= j[0] and i <= j[1]):
                num_interval = intervals.index(j)
                num_element_on_interval = i - intervals[num_interval][0]
                if (i not in index_without_henning):
                    S_after[i] += S_before_on_interval[num_interval][num_element_on_interval] * wh[
                        num_element_on_interval]
                else:
                    S_after[i] += S_before_on_interval[num_interval][num_element_on_interval]

    return S_after


def pos(BGR_signal, fps, l):
    # BGR_signal = BGR_signal.reshape(BGR_signal.shape[0], BGR_signal.shape[1])
    # BGR_signal = BGR_signal.reshape(BGR_signal.shape[0], BGR_signal.shape[1])
    # Количество кадров в исходных данных
    num_frames = len(BGR_signal)

    # Проверка допустимости размера исходных данных
    if (num_frames == 0):
        # Массив исходных данных пуст
        raise NameError('EmptyData')

    # Проверка допустимости значения fps
    if (fps < 9):
        # Недопустимое значение fps, для работы полосового фильтра требуется fps>=9
        raise NameError('WrongFPS')

    # Если длина нахлеста не задана
    if (l == None):
        # В статье Wang2017_2 использовалась длина равная 20 при fps=20, для сохранения пропорций l = fps
        l = int(fps)
    elif (l > 0):
        l = l // 1
    else:
        # Недопустимое значение длины нахлеста
        raise NameError('WrongLength')

    # Проверка допустимости размера исходных данных
    if (num_frames < l):
        # Недопустимая длина исходных данных, длина исходных данных должна быть не меньше длины нахлеста
        raise NameError('NotEnoughData')

    # Разделение исходного сигнала на каналы R,G,B
    R = BGR_signal[:, 2]
    G = BGR_signal[:, 1]
    B = BGR_signal[:, 0]

    # Функция полосовой фильтрации
    def bandpass_filter(data, lowcut, highcut):
        fs = fps  # Частота дискретизации (количество измерений сигнала в 1 сек)
        nyq = 0.5 * fs  # Частота Найквиста
        low = float(lowcut) / float(nyq)
        high = float(highcut) / float(nyq)
        order = 6.0  # Номер фильтра в scipy.signal.butter
        b, a = butter(order, [low, high], btype='band')
        bandpass = lfilter(b, a, data)
        return bandpass

    # Массив данных с которым работает метод (изначально BGR, затем преобразуем в RGB)
    RGB = np.transpose(np.array([R, G, B]))

    H = np.zeros(num_frames)

    for n in range(num_frames - l):
        m = n - l + 1
        # Массив, содержащий часть исходных данных (от m-й до n-й строки)
        C = RGB[m:n, :].T
        if m >= 0:
            # Нормализация
            mean_color = np.mean(C, axis=1)
            diag_mean_color = np.diag(mean_color)
            diag_mean_color_inv = np.linalg.inv(diag_mean_color)
            Cn = np.matmul(diag_mean_color_inv, C)

            # Матрица коэффициентов
            projection_matrix = np.array([[0, 1, -1], [-2, 1, 1]])

            S = np.matmul(projection_matrix, Cn)

            # Полосовая фильтрация (здесь S[0,:] и S[1,:] подобны Xs и Ys в методе su.chrom)
            S[0, :] = bandpass_filter(S[0, :], 0.5, 4.0)
            S[1, :] = bandpass_filter(S[1, :], 0.5, 4.0)

            # Здесь S[0,:] - S1, S[1,:] - S2 по алгоритму в Wang2017_2
            std = np.array([1, np.std(S[0, :]) / np.std(S[1, :])])
            h = np.matmul(std, S)

            # Вычисление итогового сигнала
            # Деление на np.std(h) взято с реализации в интернете
            # https://github.com/pavisj/rppg-pos/blob/master/pos_face_seg.py
            # После добавления деления на среднее отклонение всплески данных в результирующем сигнале уходят
            H[m:n] = H[m:n] + (h - np.mean(h)) / np.std(h)

    return H


def get_box_video(video, box):
    """
    Get a box from a video
    :param video: video ( T x H x W x C )
    :param box: box
    :return: box from video
    """
    return video[:, box[:, 0]:box[:, 2], box[:, 1]:box[:, 3], :]
