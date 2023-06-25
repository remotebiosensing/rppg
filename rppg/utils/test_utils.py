import os
import pandas as pd

def save_result(result_path, result, cfg):
    idx = '_'.join([cfg.model, cfg.train.dataset, cfg.test.dataset, str(cfg.img_size),
                    str(cfg.test.batch_size // cfg.test.fs)])
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if os.path.isfile(result_path + 'result.csv'):
        remaining_result = pd.read_csv(result_path + 'result.csv', index_col=0)
        if idx in remaining_result.index.tolist():
            remaining_result = remaining_result.drop(idx)
        new_result = pd.DataFrame(columns=cfg.test.metric, index=[idx])
        new_result[cfg.test.metric] = [result[-1]]
        merged_result = pd.concat([remaining_result, new_result])
        merged_result.to_csv(result_path + 'result.csv')
    else:
        new_result = pd.DataFrame(columns=cfg.test.metric, index=[idx])
        new_result[cfg.test.metric] = [result[-1]]
        new_result.to_csv(result_path + 'result.csv')
    return
