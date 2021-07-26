import numpy as np

def Deepphys_preprocess_Label(path):
    '''
    :param path: label file path
    :return: delta pulse
    '''
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
    delta_pulse = delta_label.copy() # 이거 왜 있지?
    f.close()

    # Normalize
    # part = 0
    # window = 32
    # while part < (len(delta_pulse) // window) - 1:
    #     delta_pulse[part * window:(part + 1) * window] -= np.mean(delta_pulse[part * window:(part + 1) * window])
    #     delta_pulse[part * window:(part + 1) * window] /= np.std(delta_pulse[part * window:(part + 1) * window])
    #     part += 1
    # if len(delta_pulse) % window != 0:
    #     delta_pulse[part * window:] -= np.mean(delta_pulse[part * window:])
    #     delta_pulse[part * window:] /= np.std(delta_pulse[part * window:])

    return delta_pulse

def PhysNet_preprocess_Label(path):
    '''
    :param path: label file path
    :return: wave form
    '''
    # Load input
    f = open(path, 'r')
    f_read = f.read().split('\n')
    label = ' '.join(f_read[0].split()).split()
    label = list(map(float, label))
    label = np.array(label).astype('float32')
    split_raw_label = np.zeros(((len(label) // 32), 32))
    index = 0
    for i in range(len(label) // 32):
        split_raw_label[i] = label[index:index + 32]
        index = index + 32
    f.close()

    return split_raw_label

