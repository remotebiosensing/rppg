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
from rppg.train import train_fn, val_fn, test_fn

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

    cfg = get_config("../../rppg/configs/FIT_PHYSNET_UBFC_UBFC.yaml")
    if cfg.preprocess.flag:
        preprocessing(
            dataset_root_path=cfg.data_root_path,
            preprocess_cfg=cfg.preprocess
        )

    if cfg.fit.flag:
        # load dataset

        if cfg.fit.train.flag and cfg.fit.val.flag and cfg.fit.test.flag:
            if cfg.fit.train.dataset == cfg.fit.val.dataset == cfg.fit.test.dataset:
                dataset = dataset_loader(
                    save_root_path=cfg.dataset_path,
                    model_name=cfg.fit.model,
                    dataset_name=cfg.fit.train.dataset,
                    time_length=cfg.fit.time_length,
                    overlap_interval=cfg.fit.overlap_interval,
                    img_size=cfg.fit.img_size
                )
                train_dataset, val_dataset, test_dataset = dataset_split(dataset, [0.8, 0.1, 0.1])
                train_dataset, val_dataset, test_dataset = data_loader(
                    datasets=[train_dataset, val_dataset, test_dataset],
                    batch_sizes=[cfg.fit.train.batch_size,
                                 cfg.fit.val.batch_size,
                                 cfg.fit.test.batch_size],
                    shuffles=[cfg.fit.train.shuffle,
                              cfg.fit.val.shuffle,
                              cfg.fit.test.shuffle]

                )

            elif cfg.fit.train.dataset == cfg.fit.val.dataset != cfg.fit.test.dataset:
                dataset = dataset_loader(
                    save_root_path=cfg.dataset_path,
                    model_name=cfg.fit.model,
                    dataset_name=cfg.fit.train.dataset,
                    time_length=cfg.fit.time_length,
                    overlap_interval=cfg.fit.overlap_interval,
                    img_size=cfg.fit.img_size
                )
                train_dataset, val_dataset = dataset_split(dataset, [0.8, 0.2])
                test_dataset = dataset_loader(
                    save_root_path=cfg.dataset_path,
                    model_name=cfg.fit.model,
                    dataset_name=cfg.fit.test.dataset,
                    time_length=cfg.fit.time_length,
                    overlap_interval=cfg.fit.overlap_interval,
                    img_size=cfg.fit.img_size
                )
                train_dataset, val_dataset, test_dataset = data_loader(
                    datasets=[train_dataset, val_dataset, test_dataset],
                    batch_sizes=[cfg.fit.train.batch_size,
                                 cfg.fit.val.batch_size,
                                 cfg.fit.test.batch_size],
                    shuffles=[cfg.fit.train.shuffle,
                              cfg.fit.val.shuffle,
                              cfg.fit.test.shuffle]

                )
            else:
                train_dataset = dataset_loader(
                    save_root_path=cfg.dataset_path,
                    model_name=cfg.fit.model,
                    dataset_name=cfg.fit.train.dataset,
                    time_length=cfg.fit.time_length,
                    overlap_interval=cfg.fit.overlap_interval,
                    img_size=cfg.fit.img_size
                )
                val_dataset = dataset_loader(
                    save_root_path=cfg.dataset_path,
                    model_name=cfg.fit.model,
                    dataset_name=cfg.fit.val.dataset,
                    time_length=cfg.fit.time_length,
                    overlap_interval=cfg.fit.overlap_interval,
                    img_size=cfg.fit.img_size
                )
                test_dataset = dataset_loader(
                    save_root_path=cfg.dataset_path,
                    model_name=cfg.fit.model,
                    dataset_name=cfg.fit.test.dataset,
                    time_length=cfg.fit.time_length,
                    overlap_interval=cfg.fit.overlap_interval,
                    img_size=cfg.fit.img_size
                )
                train_dataset, val_dataset, test_dataset = data_loader(
                    datasets=[train_dataset, val_dataset, test_dataset],
                    batch_sizes=[cfg.fit.train.batch_size,
                                 cfg.fit.val.batch_size,
                                 cfg.fit.test.batch_size],
                    shuffles=[cfg.fit.train.shuffle,
                              cfg.fit.val.shuffle,
                              cfg.fit.test.shuffle]

                )
        elif  cfg.fit.test.flag:
            test_data = dataset_loader(
                save_root_path=cfg.dataset_path,
                model_name=cfg.fit.model,
                dataset_name=cfg.fit.test.dataset,
                time_length=cfg.fit.time_length,
                overlap_interval=cfg.fit.overlap_interval,
                img_size=cfg.fit.img_size
            )
            test_data = data_loader(
                datasets=[test_data],
                batch_sizes=[cfg.fit.test.batch_size],
                shuffles=[cfg.fit.test.shuffle],
            )

        model = get_model(
            model_name=cfg.fit.model,
            time_length=cfg.fit.time_length)

        wandb_cfg = get_config("../../rppg/configs/WANDB_CONFG.yaml")
        if wandb_cfg.flag and cfg.fit.train.flag:
            wandb.init(project=wandb_cfg.wandb_project_name,
                       entity=wandb_cfg.wandb_entity,
                       name=cfg.fit.model +"/TRAIN_DATA" +
                            cfg.fit.train.dataset + "/TEST_DATA"+
                            cfg.fit.test.dataset + "/"+
                            str(cfg.fit.time_length) +"/" +
                            datetime.datetime.now().strftime('%m-%d%H:%M:%S'))
            wandb.config = {
                "learning_rate": cfg.fit.train.learning_rate,
                "epochs": cfg.fit.train.epochs,
                "batch_size": cfg.fit.train.batch_size
            }
            wandb.watch(model, log="all", log_freq=10)

        if cfg.fit.train.flag:
            opt = optimizer(
                model_params=model.parameters(),
                learning_rate=cfg.fit.train.learning_rate,
                optim=cfg.fit.train.optimizer)
            criterion = loss_fn(loss_name = cfg.fit.train.loss)
            lr_sch = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=cfg.fit.train.learning_rate, epochs=cfg.fit.train.epochs,
                steps_per_epoch=len(train_dataset))

            for epoch in range(cfg.fit.train.epochs):
                train_fn(
                    epoch=epoch,
                    model=model,
                    optimizer=opt,
                    criterion=criterion,
                    dataloaders=train_dataset,
                    wandb_flag=wandb_cfg.flag
                )
                if cfg.fit.val.flag and epoch % cfg.fit.val.interval == 0:
                    val_fn(
                        epoch=epoch,
                        model=model,
                        criterion=criterion,
                        dataloaders=val_dataset,
                        wandb_flag=wandb_cfg.flag
                    )
                if cfg.fit.test.flag and epoch % cfg.fit.test.interval == 0:
                    test_fn(
                        epoch=epoch,
                        model=model,
                        dataloaders=test_dataset,
                        model_name=cfg.fit.model,
                        cal_type=cfg.fit.test.cal_type,
                        metrics=cfg.fit.test.metric,
                        wandb_flag=wandb_cfg.flag
                    )
        elif cfg.fit.test.flag:
            test_fn(
                epoch=None,
                model=model,
                dataloaders=test_dataset,
                cal_type=cfg.fit.test.cal_type,
                metrics=cfg.fit.test.metric,
                wandb_flag=wandb_cfg.flag
            )
