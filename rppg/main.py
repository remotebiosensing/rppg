import os
import random
import datetime

import numpy as np
import pandas as pd
import torch
import wandb

from rppg.loss import loss_fn
from rppg.models import get_model
from rppg.optim import optimizer
from rppg.config import get_config
from rppg.dataset_loader import (dataset_loader, dataset_split, data_loader)
from rppg.preprocessing.dataset_preprocess import check_preprocessed_data
from rppg.run import run

SEED = 0

# for Reproducible model
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # if use multi-GPU
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)

generator = torch.Generator()
generator.manual_seed(SEED)

def save_result(result_path, result, cfg):
    idx = '_'.join([cfg.model, cfg.train.dataset, cfg.test.dataset, str(cfg.img_size),
                    str(cfg.test.batch_size // cfg.test.fs)])
    if os.path.exists(result_path):
        remaining_result = pd.read_csv(result_path + 'result.csv')
    else:
        new_result = pd.DataFrame(columns=cfg.test.metric,
                                  index=[idx])
        new_result[cfg.test.metric] = [result[-1]]
        print('test')
    print('test')


if __name__ == "__main__":

    fit_cfg = get_config("configs/fit.yaml")
    preprocess_cfg = get_config("configs/preprocess.yaml")
    result_save_path = 'results/csv/'

    check_preprocessed_data(fit_cfg, preprocess_cfg)

    datasets = dataset_loader(
        save_root_path=preprocess_cfg.dataset_path,
        model_name=fit_cfg.fit.model,
        dataset_name=[fit_cfg.fit.train.dataset, fit_cfg.fit.test.dataset],
        time_length=fit_cfg.fit.time_length,
        overlap_interval=fit_cfg.fit.overlap_interval,
        img_size=fit_cfg.fit.img_size,
        train_flag=fit_cfg.fit.train_flag,
        eval_flag=fit_cfg.fit.eval_flag,
        debug_flag=fit_cfg.fit.debug_flag,
        meta=fit_cfg.fit.train.meta.flag
    )

    data_loaders = data_loader(
        datasets=datasets,
        batch_size=fit_cfg.fit.batch_size,
        model_type=fit_cfg.fit.type,
        time_length=fit_cfg.fit.time_length,
        shuffle=fit_cfg.fit.train.shuffle,
    )
    model = get_model(fit_cfg.fit)

    wandb_cfg = get_config("configs/WANDB_CONFG.yaml")
    if wandb_cfg.flag and fit_cfg.fit.train_flag:
        wandb.init(project=wandb_cfg.wandb_project_name,
                   entity=wandb_cfg.wandb_entity,
                   name=fit_cfg.fit.model + "/" +
                        fit_cfg.fit.train.dataset + "/" +
                        fit_cfg.fit.test.dataset + "/" +
                        str(fit_cfg.fit.img_size) + "/" +
                        str(fit_cfg.fit.test.batch_size // fit_cfg.fit.train.fs) + "/" +
                        datetime.datetime.now().strftime('%m-%d%H:%M:%S'))
        wandb.config = {
            "learning_rate": fit_cfg.fit.train.learning_rate,
            "epochs": fit_cfg.fit.train.epochs,
            "batch_size": fit_cfg.fit.batch_size
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
        lr_sch = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=fit_cfg.fit.train.learning_rate, epochs=fit_cfg.fit.train.epochs,
            steps_per_epoch=len(datasets[0]))
    test_result = run(model, opt, lr_sch, criterion, fit_cfg, data_loaders, wandb_cfg.flag)

    save_result(result_save_path, test_result, fit_cfg.fit)


