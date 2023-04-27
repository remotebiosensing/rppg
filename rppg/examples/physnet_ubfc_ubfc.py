from rppg.config import get_config
from rppg.preprocessing.dataset_preprocess import preprocessing
from rppg.dataset_loader import (dataset_loader, dataset_split, split_data_loader)
from rppg.train import train_fn, test_fn
from models import get_model
from optim import optimizer
from loss import loss_fn
import torch
import numpy as np
import random
import os
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
    cfg = get_config("../configs/FIT_PHYSNET_UBFC_UBFC.yaml")
    if cfg.preprocess.flag:
        preprocessing()

    if cfg.fit.flag:
        # load dataset
        if cfg.fit.test.flag:
            test_data = dataset_loader()
            test_data = split_data_loader(test_data)
        elif cfg.fit.train.flag and cfg.fit.val.flag:
            if cfg.fit.train.dataset == cfg.fit.val.dataset == cfg.fit.test.dataset:
                dataset = dataset_loader()
                train_dataset, val_dataset, test_dataset = dataset_split(dataset, [0.8, 0.1, 0.1])

            elif cfg.fit.train.dataset == cfg.fit.val.dataset != cfg.fit.test.dataset:
                dataset = dataset_loader()
                train_dataset, val_dataset = dataset_split(dataset, [0.8, 0.2])
                test_dataset = dataset_loader()
            else:
                train_dataset = dataset_loader()
                val_dataset = dataset_loader()
                test_dataset = dataset_loader()

        model = get_model()
        opt = optimizer()
        criterion = loss_fn()

        if cfg.fit.train.flag:
            for epoch in range(cfg.fit.train.epochs):
                train_fn()
                if cfg.fit.val.flag and epoch % cfg.fit.val.interval == 0 :
                    test_fn()
                if cfg.fit.test.flag and epoch % cfg.fit.test.interval == 0 :
                    test_fn()
        elif cfg.fit.test.flag:
            test_fn()





