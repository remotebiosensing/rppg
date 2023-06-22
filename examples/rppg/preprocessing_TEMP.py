import random

import numpy as np
from rppg.config import get_config
from rppg.preprocessing.temp_dataset_preprocess import preprocessing

SEED = 0

# for Reproducible model
np.random.seed(SEED)
random.seed(SEED)


if __name__ == "__main__":

    cfg = get_config("../../rppg/configs/PRE_TEMP.yaml")
    if cfg.preprocess.flag:
        preprocessing(
            data_root_path=cfg.data_root_path,
            preprocess_cfg=cfg.preprocess,
            dataset_path=cfg.dataset_path
        )
