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

if __name__ == "__main__":

    cfg = get_config("../../rppg/configs/FIT_DUMMY_UBFC_UBFC.yaml")
    cfg.fit.img_size = tuple(list(map(int,(cfg.fit.img_size.split(",")))))
    if cfg.preprocess.flag:
        preprocessing(
            data_root_path=cfg.data_root_path,
            preprocess_cfg=cfg.preprocess,
            dataset_path=cfg.dataset_path
        )

        # load dataset
    datasets = dataset_loader(
        save_root_path=cfg.dataset_path,
        model_name=cfg.fit.model,
        dataset_name=[cfg.fit.train.dataset, cfg.fit.test.dataset],
        time_length=cfg.fit.time_length,
        batch_size=cfg.fit.batch_size,
        overlap_interval=cfg.fit.overlap_interval,
        img_size=cfg.fit.img_size,
        train_flag=cfg.fit.train_flag,
        eval_flag=cfg.fit.eval_flag
    )

    data_loaders = data_loader(
        datasets=datasets,
        batch_size=cfg.fit.batch_size
    )

    model = get_model(
        model_name=cfg.fit.model,
        time_length=cfg.fit.time_length,
        img_size=cfg.fit.img_size)

    wandb_cfg = get_config("../../rppg/configs/WANDB_CONFG.yaml")
    if wandb_cfg.flag and cfg.fit.train_flag:
        wandb.init(project=wandb_cfg.wandb_project_name,
                   entity=wandb_cfg.wandb_entity,
                   name=cfg.fit.model + "/TRAIN_DATA" +
                        cfg.fit.train.dataset + "/TEST_DATA" +
                        cfg.fit.test.dataset + "/" +
                        str(cfg.fit.time_length) + "/" +
                        datetime.datetime.now().strftime('%m-%d%H:%M:%S'))
        wandb.config = {
            "learning_rate": cfg.fit.train.learning_rate,
            "epochs": cfg.fit.train.epochs,
            "batch_size": cfg.fit.batch_size
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
        # lr_sch = torch.optim.lr_scheduler.OneCycleLR(
        #     opt, max_lr=cfg.fit.train.learning_rate, epochs=cfg.fit.train.epochs,
        #     steps_per_epoch=len(datasets[0]))

    run(model, opt, lr_sch, criterion, cfg.fit, data_loaders, cfg.model_path, wandb_cfg.flag)