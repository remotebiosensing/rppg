import os
import torch
import random
from torch import cuda
from torch.backends import cudnn
import numpy as np
import matplotlib.pyplot as plt
import yaml
from cnibp.preprocessing.utils.signal_utils import SignalInfoExtractor
from cnibp.preprocessing.MIMICIII_Preprocessing import get_segments_per_person
from rppg import dataset_loader
from rppg.run import test_fn
from rppg.models import get_model
from rppg.config import get_config
from rppg.dataset_loader import (dataset_loader, data_loader)
from rppg.utils.test_utils import save_sweep_result
from rppg.utils.HRV_Analyze.hrv_utils import GraphicalReport as gr

from itertools import product
from tqdm import tqdm

SEED = 0

# for Reproducible model
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# torch.use_deterministic_algorithms(mode=True, warn_only=True)
cuda.manual_seed(SEED)
cuda.manual_seed_all(SEED)  # if use multi-GPU
cuda.allow_tf32 = True
cudnn.enabled = True
cudnn.deterministic = True
cudnn.benchmark = False
cudnn.allow_tf32 = True


def get_pretrained_model_paths(pretrained_model_path, models):
    diffnorm_based_model = ['DeepPhys', 'TSCAN', 'EfficientPhys', 'BigSmall']
    diff_models = []
    cont_models = []
    # PP = []
    # PU = []
    # UP = []
    # UU = []
    for m in models:
        for root, dirs, files in os.walk(pretrained_model_path + m):
            for file in files:
                if file.endswith(".pt"):
                    model = os.path.join(root, file)
                    # if 'trainPURE_testUBFC' in file:
                    #     PU.append(model)
                    # elif 'trainUBFC_testPURE' in file:
                    #     UP.append(model)
                    # elif 'trainPURE_testPURE' in file:
                    #     UU.append(model)
                    # elif 'trainUBFC_testUBFC' in file:
                    #     PP.append(model)
                    if model.split('/')[-2] in diffnorm_based_model:
                        diff_models.append(model)
                    else:
                        cont_models.append(model)
    return diff_models, cont_models


if __name__ == "__main__":
    preset_config = get_config("../configs/model_preset.yaml")
    cfg = get_config("../configs/base_config.yaml")
    cfg.fit.train_flag = False
    cfg.fit.eval_flag = True
    cfg.fit.wandb_flag = False
    cfg.fit.debug_flag = True
    cfg.wandb.flag = False
    cfg.fit.train.batch_size = 1

    # cfg.fit.test.eval_time_length = [3, 5, 10, 20, 30]
    cfg.fit.test.eval_time_length = [10]

    models = [list(m)[0] for m in preset_config.models]
    idx = np.arange(len(models)).tolist()
    model_idx = dict(zip(models, idx))
    diff, cont = get_pretrained_model_paths(cfg.model_save_path, models)
    model_type = [diff, cont]
    hr_predictions = []
    hr_targets = []

    for mt in model_type:
        cnt = 0
        if mt == []:
            continue
        else:
            temp = None
            for pt in [mt[-2]]:
                cnt += 1
                trained_dataset = [pt.split('/')[6][5:9]]
                test_dataset = [pt.split('/')[6][14:18]]
                dataset_list = [[d, d] for d in test_dataset]
                cfg.fit.train.dataset, cfg.fit.test.dataset = dataset_list[0][0], dataset_list[0][1]

                name = pt.split('/')[-2]
                print('model name: ', name)
                m_info = preset_config.models[model_idx[name]][name]
                cfg.fit.model = m_info['model']
                cfg.fit.img_size = m_info['img_size']
                cfg.fit.type = m_info['type']
                cfg.fit.time_length = m_info['time_length']
                cfg.fit.test.batch_size = m_info['batch_size']
                cfg.preprocess.common.img_size = m_info['img_size']
                cfg.preprocess.common.type = m_info['preprocess_type']
                cfg.fit.train.dataset = trained_dataset[0]
                cfg.fit.test.dataset = test_dataset[0]

                model = get_model(cfg.fit)
                model.load_state_dict(torch.load(pt), strict=False)

                dset = dataset_loader(fit_cfg=cfg.fit, dataset_path=cfg.dataset_path)
                test_loader = data_loader(datasets=dset, fit_cfg=cfg.fit)[0]

                test_results = []
                for e in cfg.fit.test.eval_time_length:
                    result, hr_prediction, hr_target = test_fn(15, model, test_loader, cfg.fit.test, e)
                    gr(hr_target, hr_prediction).bland_altman_plot(models=models,
                                                                   title='Bland-Altman Plot',
                                                                   xlabel='Average HR of the target and {}'.format(name),
                                                                   ylabel='Difference HR between the target and {}'.format(name))
                    test_results.append(result)
                    # hr_predictions.append(hr_prediction)
                    # hr_targets.append(hr_target)
                    # test_results.append(test_fn(15, model, test_loader, cfg.fit.test.vital_type, cfg.fit.test.cal_type,
                    #                             cfg.fit.test.bpf, cfg.fit.test.metric, e))
                save_sweep_result('/home/paperc/PycharmProjects/rppg/rppg/result/csv/', test_results, cfg.fit)
                print('breakpoint')
                # if cnt == len(models):
                #     break
    # min_length = min([len(x) for x in hr_predictions])
    # hr_predictions = np.array([x[:min_length] for x in hr_predictions])
    # min_length_target = min([len(x) for x in hr_targets])
    # hr_targets = np.array([x[:min_length_target] for x in hr_targets])
    # gr(hr_targets, hr_predictions).bland_altman_plot(models=models,
    #                                                  title='Bland-Altman Plot',
    #                                                  xlabel='Average of the target and {}'.format(),
    #                                                  ylabel='Difference between the target and {}'.format())
