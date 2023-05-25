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
from rppg.utils.funcs import (get_hr, MAE, RMSE, MAPE, corr, IrrelevantPowerRatio)
# from rppg.train import train_fn, val_fn, test_fn
from rppg.MAML import MAML
# from rppg.run import test_fn
from tqdm import tqdm
from rppg.run import run

import os
import matplotlib.pyplot as plt
from rppg.utils.HR_Analyze.MMPD import BVPsignal as bvp
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
    cfg = get_config("../../rppg/configs/FINETUNE.yaml")

    dataset = dataset_loader(
        save_root_path=cfg.dataset_path,
        model_name=cfg.fit.model,
        dataset_name=[cfg.fit.train.dataset, cfg.fit.test.dataset],
        time_length=cfg.fit.time_length,
        overlap_interval=cfg.fit.overlap_interval,
        img_size=cfg.fit.img_size,
        train_flag=cfg.fit.train_flag,
        eval_flag=cfg.fit.eval_flag,
        meta=True
    )

    tasks = data_loader(datasets=dataset,
                        batch_size=cfg.fit.train.batch_size,
                        meta=True)

    model = get_model(
        model_name=cfg.fit.model,
        time_length=cfg.fit.time_length)
    if cfg.fit.pretrain.flag:
        if cfg.fit.meta.flag:
            pretrained_path = cfg.temp_path +\
                              "Meta_" + cfg.fit.model + "_" +\
                              cfg.fit.pretrain.dataset +\
                              str(cfg.fit.meta.outer_update_num) + "_" +\
                              str(cfg.fit.meta.inner_update_num) + ".pt"
        else:
            pretrained_path = cfg.fit.pretrain.path + cfg.fit.model + "_" + cfg.fit.pretrain.dataset + ".pt"
        if os.path.isfile(pretrained_path):
            model.load_state_dict(torch.load(pretrained_path))
            print("Loading pre-trained model : {}".format(pretrained_path))
        else:
            raise FileExistsError("No pre-trained model")
    else:
        print("Cold start with no pre-trained model")
    wandb_cfg = get_config("../../rppg/configs/WANDB_CONFG.yaml")

    if wandb_cfg.flag:
        wandb.init(project=wandb_cfg.wandb_project_name,
                   entity=wandb_cfg.wandb_entity,
                   name="Meta_" + cfg.fit.model + "/TRAIN_DATA:" +
                        cfg.fit.train.dataset + "/TEST_DATA:" +
                        cfg.fit.test.dataset + "/" +
                        str(cfg.fit.time_length) + "/" +
                        datetime.datetime.now().strftime('%m-%d%H:%M:%S'))
        wandb.config = {
            "learning_rate": cfg.fit.train.learning_rate,
            "epochs": cfg.fit.train.epochs,
            "batch_size": cfg.fit.batch_size
        }

    opt = None
    criterion = None
    lr_sch = None
    if cfg.fit.train_flag:
        opt = optimizer(
            model_params=model.parameters(),
            learning_rate=cfg.fit.train.learning_rate,
            optim=cfg.fit.train.optimizer)
        criterion = loss_fn(loss_name=cfg.fit.train.loss)

    run(model, opt, lr_sch, criterion, cfg.fit, tasks[0], cfg.model_path, wandb_cfg.flag)
