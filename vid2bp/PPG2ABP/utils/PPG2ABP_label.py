from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pickle

length = 352


def prepareData(model, X_train, X_val, X_test, Y_train, Y_val, Y_test):
    """
    Prepares data for 2nd stage training

    Arguments:
        mdl {keras.model} -- keras model
        X_train {array} -- X train
        X_val {array} -- X val
        X_test {array} -- X test
        Y_train {array} -- Y train
        Y_val {array} -- Y val
        Y_test {array} -- Y test

    Returns:
        tuple -- tuple of X_train, X_val and X_test for 2nd stage training
    """

    X2_train = []

    X2_val = []

    X2_test = []

    YPs = model.predict(X_train)

    for i in tqdm(range(len(X_train))):
        X2_train.append(np.array(YPs[i]))

    YPs = model.predict(X_val)

    for i in tqdm(range(len(X_val))):
        X2_val.append(np.array(YPs[i]))

    YPs = model.predict(X_test)

    for i in tqdm(range(len(X_test))):
        X2_test.append(np.array(YPs[i]))

    X2_train = np.array(X2_train)

    X2_val = np.array(X2_val)

    X2_test = np.array(X2_test)

    return (X2_train, X2_val, X2_test)


def prepareDataDS(mdl, X):
    """
    Prepares data for 2nd stage training in the deep supervised pipeline

    Arguments:
        mdl {keras.model} -- keras model
        X {array} -- array being X train or X val

    Returns:
        X {array} -- suitable X for 2nd stage training
    """

    X2 = []

    YPs = mdl.predict(X)

    for i in tqdm(range(len(X)), desc='Preparing Data for DS'):
        X2.append(np.array(YPs[0][i]))

    X2 = np.array(X2)

    return X2


def prepareLabel(Y):
    def approximate(inp, w_len):
        op = []
        for i in range(0, len(inp), w_len):
            op.append(np.mean(inp[i:i + w_len]))

        return np.array(op)

    out = {}
    out['out'] = []
    out['level1'] = []
    out['level2'] = []
    out['level3'] = []
    out['level4'] = []

    for y in tqdm(Y, desc='Preparing Label for DS'):
        cA1 = approximate(np.array(y).reshape(length), 2)
        cA2 = approximate(np.array(y).reshape(length), 4)
        cA3 = approximate(np.array(y).reshape(length), 8)
        cA4 = approximate(np.array(y).reshape(length), 16)

        out['out'].append(np.array(y.reshape(length, 1)))
        out['level1'].append(np.array(cA1.reshape(length // 2, 1)))
        out['level2'].append(np.array(cA2.reshape(length // 4, 1)))
        out['level3'].append(np.array(cA3.reshape(length // 8, 1)))
        out['level4'].append(np.array(cA4.reshape(length // 16, 1)))

    out['out'] = np.array(out['out'])  # converting to numpy array
    out['level1'] = np.array(out['level1'])
    out['level2'] = np.array(out['level2'])
    out['level3'] = np.array(out['level3'])
    out['level4'] = np.array(out['level4'])

    return out
