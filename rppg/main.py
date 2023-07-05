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
from rppg.utils.test_utils import save_single_result

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

    fit_cfg = get_config("configs/fit.yaml")
    preprocess_cfg = get_config("configs/preprocess.yaml")
    result_save_path = 'result/csv/'

    check_preprocessed_data(fit_cfg, preprocess_cfg)

    datasets = dataset_loader(fit_cfg=fit_cfg.fit, pre_cfg=preprocess_cfg)

    data_loaders = data_loader(datasets=datasets, fit_cfg=fit_cfg.fit)

    model = get_model(fit_cfg.fit)

    if fit_cfg.wandb.flag and fit_cfg.fit.train_flag:
        wandb.init(project=fit_cfg.wandb.project_name,
                   entity=fit_cfg.wandb.entity,
                   name=fit_cfg.fit.model + "/" +
                        fit_cfg.fit.train.dataset + "/" +
                        fit_cfg.fit.test.dataset + "/" +
                        str(fit_cfg.fit.img_size) + "/" +
                        str(fit_cfg.fit.test.batch_size // fit_cfg.fit.train.fs) + "/" +
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
        lr_sch = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=fit_cfg.fit.train.learning_rate, epochs=fit_cfg.fit.train.epochs,
            steps_per_epoch=len(datasets[0]))
    test_result = run(model, False, opt, lr_sch, criterion, fit_cfg, data_loaders, wandb_flag=fit_cfg.wandb.flag)

    save_single_result(result_save_path, test_result, fit_cfg.fit)

    sys.exit(0)
