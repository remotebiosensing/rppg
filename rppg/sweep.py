import sys
import random
import datetime

import wandb
import numpy as np
import torch
import torch.cuda as cuda
import torch.backends.cudnn as cudnn

from loss import loss_fn
from models import get_model
from optim import optimizer
from config import get_config
from dataset_loader import (dataset_loader, dataset_split, data_loader)
from preprocessing.dataset_preprocess import check_preprocessed_data
from run import run
from utils.test_utils import save_sweep_result
from itertools import product

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

# generator = torch.Generator()
# generator.manual_seed(SEED)

if __name__ == "__main__":

    result_save_path = 'result/csv/'

    diffnorm_based_model = ['DeepPhys', 'TSCAN', 'EfficientPhys', 'BigSmall']
    non_dnn_model = ['CHROM', 'GREEN', 'POS', 'LGI', 'PCA', 'SSR', 'ICA']

    preset_cfg = get_config("configs/model_preset.yaml")
    models = [list(m)[0] for m in preset_cfg.models]
    test_eval_time_length = [3, 5, 10, 20, 30]  # in seconds
    dataset_list = ['UBFC', 'PURE']
    list_product = list(product(dataset_list, repeat=2))
    datasets = [list(x) for x in list_product][:1]
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
        cfg = get_config("configs/base_config.yaml")
        if cfg.fit.debug_flag is True:
            print("Debug mode is on.\n No wandb logging & Not saving csv and model")
            cfg.wandb.flag = False
            cfg.fit.model_save_flag = False
        cfg.wandb.flag = not cfg.fit.debug_flag
        cfg.fit.train.dataset = cfg.preprocess.train_dataset.name = d[0]
        cfg.fit.test.dataset = cfg.preprocess.test_dataset.name = d[1]


        for m, i, m_t, p_t, t, b, l, loss, o in zip(model_name, img_size, model_type, preprocess_type, time_length,
                                                    batch_size,
                                                    learning_rate, losses, opts):
            cfg.fit.model, cfg.fit.img_size, cfg.fit.type, cfg.preprocess.common.type = m, i, m_t, p_t
            cfg.fit.time_length, cfg.fit.train.learning_rate = t, l
            cfg.fit.train.batch_size, cfg.fit.test.batch_size = b, b
            cfg.fit.train.loss, cfg.fit.train.optimizer = loss, o

            if cfg.fit.model in non_dnn_model:
                cfg.fit.train_flag = False
                # cfg.fit.type = 'CONT_RAW'
            check_preprocessed_data(cfg)
            dset = dataset_loader(fit_cfg=cfg.fit, dataset_path=cfg.dataset_path)
            data_loaders = data_loader(datasets=dset, fit_cfg=cfg.fit)
            cfg.fit.test.eval_time_length = test_eval_time_length

            model = get_model(cfg.fit)

            if cfg.wandb.flag and cfg.fit.train_flag:
                wandb.init(project=cfg.wandb.project_name,
                           entity=cfg.wandb.entity,
                           name=cfg.fit.model + "/" +
                                cfg.fit.train.dataset + "/" +
                                cfg.fit.test.dataset + "/" +
                                str(cfg.fit.img_size) + "/" +
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
                # lr_sch = torch.optim.lr_scheduler.OneCycleLR(
                #     opt, max_lr=0.1, epochs=cfg.fit.train.epochs,
                #     steps_per_epoch=len(datasets[0]))
                # lr_sch = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5)
            test_result = run(model, True, opt, lr_sch, criterion, cfg, data_loaders)
            if not cfg.fit.debug_flag:
                save_sweep_result(result_save_path, test_result, cfg.fit)

            wandb.finish()

    sys.exit(0)
