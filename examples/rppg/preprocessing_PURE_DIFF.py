import random

import numpy as np
import torch
from rppg.loss import loss_fn
from rppg.config import get_config
from rppg.preprocessing.dataset_preprocess import preprocessing

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

    cfg = get_config("../../rppg/configs/PRE_DIFF_PURE.yaml")
    if cfg.preprocess.flag:
        preprocessing(
            data_root_path=cfg.data_root_path,
            preprocess_cfg=cfg.preprocess,
            dataset_path=cfg.dataset_path
        )
