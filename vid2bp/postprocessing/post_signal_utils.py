import numpy as np
from sklearn import preprocessing

def shape_scaler(input_sig):
    input_sig = np.reshape(input_sig, (-1, 1))
    scaled_output = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(input_sig)
    return np.squeeze(scaled_output)


def feature_combiner(corr, pred_s, pred_d):
    scaled_corr = shape_scaler(corr)
    amplitude = (pred_s.detach().cpu()[0] - pred_d.detach().cpu()[0]).numpy()
    height = pred_d.detach().cpu()[0].numpy()

    abp_prediction = scaled_corr * amplitude + height

    return abp_prediction
