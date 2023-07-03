import os
import pandas as pd


def save_single_result(result_path, result, cfg):
    idx = '_'.join([cfg.model, cfg.train.dataset, cfg.test.dataset, str(cfg.img_size),
                    str(cfg.train.epochs), str(cfg.test.eval_time_length)])
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if os.path.isfile(result_path + 'result.csv'):
        remaining_result = pd.read_csv(result_path + 'result.csv', index_col=0)
        if idx in remaining_result.index.tolist():
            print("Warning: result already exists, overwriting...")
            # remaining_result.drop_duplicates(subset)
            remaining_result = remaining_result.drop(idx)
        new_result = pd.DataFrame(columns=cfg.test.metric, index=[idx])
        new_result[cfg.test.metric] = [result]
        merged_result = pd.concat([remaining_result, new_result]).sort_index()
        merged_result.to_csv(result_path + 'result.csv')
    else:
        new_result = pd.DataFrame(columns=cfg.test.metric, index=[idx])
        new_result[cfg.test.metric] = [result]
        new_result.to_csv(result_path + 'result.csv')

def save_sweep_result(result_path, results, cfg):
    idxs = []
    for et in cfg.test.eval_time_length:
        idxs.append('_'.join([cfg.model, cfg.train.dataset, cfg.test.dataset, str(cfg.img_size),
                              str(cfg.train.epochs), str(et)]))
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if os.path.isfile(result_path + 'result.csv'):
        remaining_result = pd.read_csv(result_path + 'result.csv', index_col=0)
        for idx in idxs:
            if idx in remaining_result.index.tolist():
                print("Notice: Prior result already exists, overwriting...")
                remaining_result = remaining_result.drop(idx)
        new_result = pd.DataFrame(columns=cfg.test.metric, index=idxs)
        new_result[cfg.test.metric] = results
        merged_result = pd.concat([remaining_result, new_result]).sort_index()
        merged_result.to_csv(result_path + 'result.csv')
    else:
        new_result = pd.DataFrame(columns=cfg.test.metric, index=idxs)
        new_result[cfg.test.metric] = results
        new_result.to_csv(result_path + 'result.csv')




    print('sweep')
