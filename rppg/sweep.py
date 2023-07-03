import sys
import random
import datetime

import numpy as np
import torch
import wandb

from rppg.loss import loss_fn
from rppg.models import get_model
from rppg.optim import optimizer
from rppg.config import get_config
from rppg.dataset_loader import (dataset_loader, dataset_split, data_loader)
from rppg.preprocessing.dataset_preprocess import check_preprocessed_data
from rppg.run import run
from rppg.utils.test_utils import save_sweep_result

SEED = 0

# for Reproducible model
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)

generator = torch.Generator()
generator.manual_seed(SEED)

if __name__ == "__main__":

    result_save_path = 'result/csv/'

    diffnorm_based_model = ['DeepPhys', 'TSCAN', 'EfficientPhys', 'BigSmall']

    preset_cfg = get_config("configs/model_preset.yaml")
    models = [list(m)[0] for m in preset_cfg.models]
    test_eval_time_length = [3, 5, 10, 20, 30]  # in seconds
    datasets = [['UBFC', 'PURE']]  # ,['PURE', 'PURE'],['UBFC', 'UBFC'],['PURE', 'UBFC']]

    model_name = []
    model_type = []
    preprocess_type = []
    img_size = []
    time_length = []
    batch_size = []
    learning_rate = []
    opts = []
    losses = []

    for m, name in zip(preset_cfg.models, models):
        print(m, name)
        model_name.append(m[name]['model'])
        model_type.append(m[name]['type'])
        preprocess_type.append(m[name]['preprocess_type'])
        time_length.append(m[name]['time_length'])
        batch_size.append(m[name]['batch_size'])
        learning_rate.append(m[name]['learning_rate'])
        img_size.append(m[name]['img_size'])
        opts.append(m[name]['optimizer'])
        losses.append(m[name]['loss'])

    for d in datasets:
        fit_cfg = get_config("configs/fit.yaml")
        preprocess_cfg = get_config("configs/preprocess.yaml")
        fit_cfg.fit.train.dataset, fit_cfg.fit.test.dataset = d[0], d[1]

        for m, i, m_t, p_t, t, b, l, loss, o in zip(model_name, img_size, model_type, preprocess_type, time_length,
                                                    batch_size,
                                                    learning_rate, losses, opts):
            fit_cfg.fit.model, fit_cfg.fit.img_size, fit_cfg.fit.type, preprocess_cfg.dataset.type = m, i, m_t, p_t
            fit_cfg.fit.time_length, fit_cfg.fit.train.learning_rate = t, l
            fit_cfg.fit.train.batch_size, fit_cfg.fit.test.batch_size = b, b
            fit_cfg.fit.train.loss, fit_cfg.fit.train.optimizer = loss, o

            check_preprocessed_data(fit_cfg, preprocess_cfg)
            dset = dataset_loader(fit_cfg=fit_cfg.fit, pre_cfg=preprocess_cfg)
            data_loaders = data_loader(datasets=dset, fit_cfg=fit_cfg.fit)
            fit_cfg.fit.test.eval_time_length = test_eval_time_length

            model = get_model(fit_cfg.fit)

            if fit_cfg.wandb.flag and fit_cfg.fit.train_flag:
                wandb.init(project=fit_cfg.wandb.project_name,
                           entity=fit_cfg.wandb.entity,
                           name=fit_cfg.fit.model + "/" +
                                fit_cfg.fit.train.dataset + "/" +
                                fit_cfg.fit.test.dataset + "/" +
                                str(fit_cfg.fit.img_size) + "/" +
                                datetime.datetime.now().strftime('%m-%d%H:%M:%S'))
                wandb.config = {
                    "learning_rate": fit_cfg.fit.train.learning_rate,
                    "epochs": fit_cfg.fit.train.epochs,
                    "train_batch_size": fit_cfg.fit.train.batch_size,
                    "test_batch_size": fit_cfg.fit.test.batch_size
                }
                wandb.watch(model, log="all", log_freq=10)

            opt = None
            criterion = None
            lr_sch = None
            if fit_cfg.fit.train_flag:
                opt = optimizer(
                    model_params=model.parameters(),
                    learning_rate=fit_cfg.fit.train.learning_rate,
                    optim=fit_cfg.fit.train.optimizer)
                criterion = loss_fn(loss_name=fit_cfg.fit.train.loss)
                # lr_sch = torch.optim.lr_scheduler.OneCycleLR(
                #     opt, max_lr=fit_cfg.fit.train.learning_rate, epochs=fit_cfg.fit.train.epochs,
                #     steps_per_epoch=len(datasets[0]))
            test_result = run(model, True, opt, lr_sch, criterion, fit_cfg, data_loaders, fit_cfg.wandb.flag)
            save_sweep_result(result_save_path, test_result, fit_cfg.fit)
            # save_single_result(result_save_path, test_result, fit_cfg.fit)

            wandb.finish()

    sys.exit(0)
