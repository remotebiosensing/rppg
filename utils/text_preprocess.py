import numpy as np
from sklearn.preprocessing import minmax_scale

def Deepphys_preprocess_Label(path):
    '''
    :param path: label file path
    :return: delta pulse
    '''
    # TODO : need to check length with video frames
    # TODO : need to implement piecewise cubic Hermite interpolation
    # Load input
    f = open(path, 'r')
    f_read = f.read().split('\n')
    label = ' '.join(f_read[0].split()).split()
    label = list(map(float, label))

    delta_label = []
    for i in range(len(label) - 1):
        delta_label.append(label[i + 1] - label[i])
    delta_label -= np.mean(delta_label)
    delta_label /= np.std(delta_label)
    delta_label = np.array(delta_label).astype('float32')
    delta_pulse = delta_label.copy()  # 이거 왜 있지?
    f.close()

    return delta_pulse


def PhysNet_preprocess_Label(path):
    '''
    :param path: label file path
    :return: wave form
    '''
    set = 64
    div = 64
    # Load input
    f = open(path, 'r')
    f_read = f.read().split('\n')
    label = ' '.join(f_read[0].split()).split()
    label = list(map(float, label))
    label = np.array(label).astype('float32')
    split_raw_label = np.zeros(((len(label) - div + 1), div))
    index = 0
    for i in range(0,(len(label) - div + 1),div):
        split_raw_label[i] = label[i:i + div]
        # index = index + div
    f.close()

    return split_raw_label

def GCN_preprocess_Label(path,sliding_window_stride):
    '''
    :param path: label file path
    :return: wave form
    '''

    div = 256
    stride = sliding_window_stride
    # Load input
    f = open(path, 'r')
    f_read = f.read().split('\n')
    label = ' '.join(f_read[0].split()).split()
    label = list(map(float, label))
    label = np.array(label).astype('float32')
    num_maps = int((len(label) - div)/stride + 1)
    split_raw_label = np.zeros((num_maps, div))
    index = 0
    for i in range(0,num_maps,stride):
        split_raw_label[i] = label[index:index + div]
        index = index + stride
    f.close()

    return split_raw_label

def Axis_preprocess_Label(path,sliding_window_stride,num_frames,clip_size = 256):
    '''
    :param path: label file path
    :return: wave form
    '''

    # div = 256
    # stride = num_maps
    # Load input
    ext = path.split('.')[-1]


    f = open(path, 'r')

    f_read = f.read().split('\n')
    if ext == 'txt':
        label = ' '.join(f_read[0].split()).split()
        label = list(map(float, label))


    elif ext == 'csv':
        label = f_read[1:]
        label = [float(txt) for txt in label if txt != '']

    label = np.array(label).astype('float32')
    label = np.resize(label,num_frames)
    # print(path + str(len(label))+ "  " + str(num_maps)+"  "+str(clip_size) +"  " + str(sliding_window_stride) + "  " + str(num_frames))
    # print(num_maps)
    num_maps = int((num_frames - clip_size) / sliding_window_stride + 1)
    split_raw_label = np.zeros((num_maps, clip_size))
    index = 0
    for start_frame_index in range(0, num_frames,sliding_window_stride ):
        end_frame_index = start_frame_index + clip_size
        if end_frame_index > num_frames:
            break
        split_raw_label[index,:] = minmax_scale(label[start_frame_index:end_frame_index],axis=0,copy=True)*2-1#label[start_frame_index:end_frame_index]
        index += 1
    f.close()

    return split_raw_label