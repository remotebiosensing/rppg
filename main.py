import time
import datetime
from dataset.dataset_loader import dataset_loader
from torch.utils.data import DataLoader,random_split
from utils.dataset_preprocess import preprocessing

if False:
    start_time = time.time()
    preprocessing(save_root_path="/media/hdd1/dy_dataset/",
                  data_root_path="/media/hdd1/",
                  dataset_name="UBFC",
                  train_ratio=0.8)
    print("preprocessing time \t: ", datetime.timedelta(seconds=time.time() - start_time))

start_time = time.time()
train_dataset = dataset_loader(save_root_path="/media/hdd1/dy_dataset/",
                               dataset_name="UBFC",
                               option="train")
print("load train hpy time \t: ", datetime.timedelta(seconds=time.time() - start_time))

start_time = time.time()
test_dataset = dataset_loader(save_root_path="/media/hdd1/dy_dataset/",
                              dataset_name="UBFC",
                              option="test")
print("load test hpy time \t: ", datetime.timedelta(seconds=time.time() - start_time))
