import random

import numpy as np
import torch
import wandb
import datetime
from rppg.loss import loss_fn
from rppg.models import get_model
from rppg.optim import optimizer
from rppg.config import get_config
from rppg.dataset_loader import (dataset_loader, dataset_split, data_loader)
from rppg.preprocessing.dataset_preprocess import preprocessing
# from rppg.train import train_fn, val_fn, test_fn
from rppg.MAML import MAML
from tqdm import tqdm
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

if __name__ == "__main__":

    cfg = get_config("../../rppg/configs/FIT_METAPHYS_MMPD.yaml")
    if cfg.preprocess.flag:
        preprocessing(
            dataset_root_path=cfg.data_root_path,
            preprocess_cfg=cfg.preprocess
        )


    # load dataset
    if cfg.fit.meta.flag:
        dataset = dataset_loader(
            save_root_path=cfg.dataset_path,
            model_name=cfg.fit.model,
            dataset_name=[cfg.fit.train.dataset, cfg.fit.test.dataset],
            time_length=cfg.fit.time_length,
            overlap_interval=cfg.fit.overlap_interval,
            img_size=cfg.fit.img_size,
            train_flag=cfg.fit.train_flag,
            eval_flag=cfg.fit.eval_flag,
            meta=cfg.fit.meta.flag
            )

        tasks = data_loader(datasets=dataset,
                              batch_size=cfg.fit.train.batch_size,
                              meta=cfg.fit.meta.flag)

    model = get_model(
        model_name=cfg.fit.model,
        time_length=cfg.fit.time_length)

    if cfg.fit.meta.flag:
        meta = MAML(model=model,
                    inner_optim=cfg.fit.meta.inner_optim,
                    outer_optim=cfg.fit.meta.outer_optim,
                    inner_loss=cfg.fit.meta.inner_loss,
                    outer_loss=cfg.fit.meta.outer_loss,
                    inner_lr=cfg.fit.meta.inner_lr,
                    outer_lr=cfg.fit.meta.outer_lr,
                    num_updates=5)

        for epoch in range(cfg.fit.train.epochs):
            meta.meta_update(tasks, epoch)

            # adaptation = False
            # if adaptation:
            #     for train in tasks[20:]:
            #         individual_model = meta.inner_update(tasks[:])





