import preprocessing.utils.signal_utils as su


def feature_combiner(corr, pred_s, pred_d):
    scaled_corr = su.scaler(corr)
    amplitude = (pred_s.detach().cpu()[0] - pred_d.detach().cpu()[0]).numpy()
    height = pred_d.detach().cpu()[0].numpy()

    abp_prediction = scaled_corr * amplitude + height

    return abp_prediction
