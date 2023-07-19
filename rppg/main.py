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

    cfg = get_config("configs/base_config.yaml")
    # preprocess_cfg = get_config("rppg/configs/preprocess.yaml")
    result_save_path = 'result/csv/'

    check_preprocessed_data(cfg)

    datasets = dataset_loader(fit_cfg=cfg.fit, dataset_path=cfg.dataset_path)

    data_loaders = data_loader(datasets=datasets, fit_cfg=cfg.fit)

    model = get_model(cfg.fit)

    if cfg.wandb.flag and cfg.fit.train_flag:
        wandb.init(project=cfg.wandb.project_name,
                   entity=cfg.wandb.entity,
                   name=cfg.fit.model + "/" +
                        cfg.fit.train.dataset + "/" +
                        cfg.fit.test.dataset + "/" +
                        str(cfg.fit.img_size) + "/" +
                        str(cfg.fit.test.batch_size // cfg.fit.train.fs) + "/" +
                        datetime.datetime.now().strftime('%m-%d%H:%M:%S'))
        wandb.config = {
            "learning_rate": cfg.fit.train.learning_rate,
            "epochs": cfg.fit.train.epochs,
            "train_batch_size": cfg.fit.train.batch_size,
            "test_batch_size": cfg.fit.test.batch_size
        }
        wandb.watch(model, log="all", log_freq=10)

    opt = None
    criterion = None
    lr_sch = None
    if cfg.fit.train_flag:
        opt = optimizer(
            model_params=model.parameters(),
            learning_rate=cfg.fit.train.learning_rate,
            optim=cfg.fit.train.optimizer)
        criterion = loss_fn(loss_name=cfg.fit.train.loss)
        lr_sch = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=cfg.fit.train.learning_rate, epochs=cfg.fit.train.epochs,
            steps_per_epoch=len(datasets[0]))
    test_result = run(model, False, opt, lr_sch, criterion, cfg, data_loaders)

    save_single_result(result_save_path, test_result, cfg.fit)

    sys.exit(0)
