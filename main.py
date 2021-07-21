import time
import datetime
from utils.dataset_preprocess import preprocessing

start_time = time.time()
preprocessing()
print(datetime.timedelta(seconds=time.time() - start_time))

