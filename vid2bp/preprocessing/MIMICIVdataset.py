import gzip
import os
import numpy as np
import pandas as pd

mimiciv_path = '/home/paperc/PycharmProjects/dataset/physionet.org/files/mimiciv/2.0/'

icu_chartevents = 'icu/chartevents.csv.gz'
icu_datetimeevent = 'icu/datetimeevents.csv.gz'
icu_d_items = 'icu/d_items.csv.gz'
icu_icustays = 'icu/icustays.csv.gz'
icu_ingredientevents = 'icu/ingredientevents.csv.gz'
icu_inputevents = 'icu/inputevents.csv.gz'
icu_outputevents = 'icu/outputevents.csv.gz'
icu_procedureevents = 'icu/procedureevents.csv.gz'

hosp_emar_details = 'hosp/emar_details.csv.gz'

# how to extract abp from mimiciv



# f = gzip.open(mimiciv_path+test_data, 'r')
# file_content = f.read()
# f.close()

'''
icu/d_items.csv.gz
- itemid, label
- 220050 Arterial Blood pressure systolic
- 220051 Arterial Blood pressure diastolic
- 220052 Arterial Blood pressure mean
'''
with gzip.open(mimiciv_path + icu_inputevents) as f:

    features_train = pd.read_csv(f)

features_train.head()
