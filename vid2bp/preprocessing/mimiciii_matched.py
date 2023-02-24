import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder as le
from sklearn.preprocessing import OneHotEncoder as ohe


# patients_info = pd.read_csv('/hdd/hdd1/dataset/mimiciiisubset/records/PATIENTS.csv', names=['SUBJECT_ID', 'GENDER'])
# https://growthj.link/python-%EC%9B%90-%ED%95%AB-%EC%9D%B8%EC%BD%94%EB%94%A9one-hot-encoding-%EC%A0%95%EB%A6%AC-get_dummies/
def get_patients_info(ge: int, pid: list, pid_list: list):
    gender_le = le()
    ethnicity_le = le()
    diagnosis_le = le()
    gender_ohe = ohe()
    ethnicity_ohe = ohe()
    info_path = '/hdd/hdd1/dataset/mimiciiisubset/records/PATIENTS.csv'
    admission_path = '/hdd/hdd1/dataset/mimiciiisubset/records/ADMISSIONS.csv'
    patients_df = pd.read_csv(info_path, usecols=(1, 2))
    # admission_df = pd.read_csv(admission_path, usecols=(1, 2, 6, 13, 16))
    admission_df = pd.read_csv(admission_path, usecols=(1, 2, 13, 16))
    # merged_df = patients_df + addmission_df
    merged_df = pd.merge(patients_df, admission_df, how='outer', on='SUBJECT_ID')
    # sorting columns of merged_df
    merged_df = merged_df[['SUBJECT_ID', 'HADM_ID', 'GENDER', 'ETHNICITY', 'DIAGNOSIS']]
    # extracting patient data from mimic_iii_clinical_database with mimic_iii_waveform_database
    merged_df = merged_df[merged_df['SUBJECT_ID'].isin(pid)]
    # temp_df = pd.DataFrame({'SUBJECT_ID': np.squeeze(np.split(np.array(pid_list), 2, axis=1)[0]),
    #                         'PATH': np.squeeze(np.split(np.array(pid_list), 2, axis=1)[1])},dtype=('int64','string'))
    # total_df = pd.merge(merged_df, temp_df, how='outer',on='SUBJECT_ID')
    # eth_list = list(merged_df['ETHNICITY'].value_counts)

    # df_dict = merged_df.to_dict()
    # dic_val = df_dict['ETHNICITY'].values()
    # dic_list = set(dic_val)
    # eth_group = list(set(dic_val))

    # ETHNICITY
    eth = {'WHITE': 'WHITE',
           'BLACK|AFRICAN': 'BLACK',
           'ASIAN': 'ASIAN',
           'HISPANIC|LATINO': 'HISPANIC',
           'UNKNOWN|PATIENT|OTHER|UNABLE|PORTUGUESE|MULTI|INDIAN|NATIVE': 'OTHERS & UNKNOWN'}
    for e in eth:
        merged_df.loc[merged_df['ETHNICITY'].str.contains(e), 'ETHNICITY'] = eth[e]

    # merged_df.loc[merged_df['ETHNICITY'].str.contains('BLACK'), 'ETHNICITY'] = 'BLACK'
    # merged_df.loc[merged_df['ETHNICITY'].str.contains('ASIAN'), 'ETHNICITY'] = 'ASIAN'
    # merged_df.loc[merged_df['ETHNICITY'].str.contains('HISPANIC|LATINO'), 'ETHNICITY'] = 'HISPANIC'
    # merged_df.loc[merged_df['ETHNICITY'].str.contains(
    #     'UNKNOWN|PATIENT|OTHER|UNABLE|PORTUGUESE|MULTI|INDIAN|NATIVE'), 'ETHNICITY'] = 'OTHERS & UNKNOWN'

    # DIAGNOSIS
    heart_brain_disease = {'CONGESTIVE HEART FAILURE|HEART FAILURE|CARDIAC INSUFFICIENCY': 'HEART FAILURE',  # 심부전
                           'HYPERTENSIVE|HYPERTENSION': 'HYPERTENSION',  # 고혈압
                           'DISSECTION|AORTIC DISSECTION': 'AORTIC DISSECTION',  # 대동맥 박리
                           'AORTIC STENOSIS': 'AORTIC STENOSIS',  # 대동맥 협착
                           'ANGINA PECTORIS|ANGINA': 'ANGINA',  # 협심증 : 심장에 혈액을 공급하는 혈관인 관상 동맥이 동맥 경화증으로 좁아져서 생기는 질환
                           'ARRHYTHMIA': 'ARRHYTHMIA',  # 부정맥 : 심장이 정상적으로 뛰지 않는 것(빈맥증, 서맥증)
                           'HYPOTENSION': 'HYPOTENSION',  # 고혈압
                           'STROKE|CEREBRAL INFARCTION|CEREBRAL HEMORRHAGE': 'STROKE',  # 뇌졸중
                           'STEMI|MYOCARDIAC INFARCT|MYOCARDIAL INFARCT|MYOCARDIAL INFARCTION|CARDIAL INFARCTION|CARDIAC INFARCTION|MYOCARDIAL INFARCTION': 'CARDIAC INFARCTION',  # 심근경색
                           'ARTERIOSCLEROSIS': 'ARTERIOSCLEROSIS',  # 동맥 경화증
                           'VALVE|VALVULAR|VALVE DISEASE|VALVULAR DISEASE|VALVULAR HEART': 'VALVULAR DISEASE',  # 판막증
                           'HEART ATTACK|CARDIAC ARREST': 'CARDIAC ARREST',  # 심장마비
                           'CORONARY|CORONARY ARTERY|CORONARY ARTERY BYPASS': 'CORONARY ARTERY DISEASE',  # 관상동맥 질병
                           'TACHYCARDIA|VENTRICULAR TACHYCARDIA': 'VENTRICULAR TACHYCARDIA',  # 심실빠른맥
                           'CARDIOMYOPATHY|CMS': 'CARDIOMYOPATHY',  # 심근증
                           'CHEST PAIN': 'CHEST PAIN',  # 흉통
                           # 'STEMI': 'STEMI'  # ST분절 상승 심근경색
                           }
    sorted_hb_disease = sorted(list(heart_brain_disease.keys()), key=len, reverse=True)
    for h in sorted_hb_disease:
        merged_df.loc[merged_df['DIAGNOSIS'].str.contains(h), 'reDIAGNOSIS'] = heart_brain_disease[h]  # 1650
    merged_df.loc[~merged_df['DIAGNOSIS'].str.contains(
        '|'.join(list(heart_brain_disease.keys()))), 'reDIAGNOSIS'] = 'NON CARDIAC DISEASE'

    # merged_df['DIAGNOSIS'].str.

    # dic_dia_val = df_dict['DIAGNOSIS'].values()

    gender_le.fit(list(set((merged_df['GENDER'].to_dict()).values())))
    ethnicity_le.fit(list(set((merged_df['ETHNICITY'].to_dict()).values())))
    diagnosis_le.fit(list(set((merged_df['reDIAGNOSIS'].to_dict()).values())))
    gender_label = gender_le.transform(merged_df['GENDER'])
    ethnicity_label = ethnicity_le.transform(merged_df['ETHNICITY'])
    dia_label = diagnosis_le.transform(merged_df['reDIAGNOSIS'])
    merged_df['GENDER_LABEL'] = gender_label
    merged_df['ETHNICITY_LABEL'] = ethnicity_label
    merged_df['DIAGNOSIS_LABEL'] = dia_label
    gender_ohe.fit(gender_label.reshape(-1,1))
    gender_ohe_label = gender_ohe.transform(gender_label.reshape(-1,1))
    ethnicity_ohe.fit(ethnicity_label.reshape(-1,1))
    ethnicity_ohe_label = ethnicity_ohe.transform(ethnicity_label.reshape(-1,1))

    ge_ohe_list = np.hstack((gender_ohe_label.toarray(), ethnicity_ohe_label.toarray()))

    label_df = merged_df[['SUBJECT_ID', 'HADM_ID', 'GENDER_LABEL', 'ETHNICITY_LABEL', 'DIAGNOSIS_LABEL']]

    male_info = {}
    female_info = {}
    total_info_list = [male_info, female_info]
    male_df = label_df.loc[(label_df['GENDER_LABEL'] == 1)]
    female_df = label_df.loc[(label_df['GENDER_LABEL'] == 0)]
    # neonates_df = merged_df.loc[(merged_df['ADMISSION_TYPE'] == 'NEWBORN')]
    gender = [male_df, female_df]
    # pid_dict = {}
    # for p in pid_list:
    #     pid_dict[p[0]] = p[-1]
    for g, l in zip(gender, total_info_list):

        patients_id = g['SUBJECT_ID'].values
        patients_hadm = g['HADM_ID'].values
        patients_gender = g['GENDER_LABEL'].values
        patients_ethnicity = g['ETHNICITY_LABEL'].values
        patients_diagnosis = g['DIAGNOSIS_LABEL'].values
        # patients_expire_flag = g['EXPIRE_FLAG'].values

        for idx, (id, hadm, gen, eth, diag, geohe) in enumerate(
                zip(patients_id, patients_hadm, patients_gender, patients_ethnicity, patients_diagnosis, ge_ohe_list)):
            if l.get(str(format(id, '05'))) is None:
                try:
                    # l[str(format(id, '05'))] = [id, hadm, gen, eth, diag, pid_dict[id]]
                    l[str(format(id, '05'))] = [id, hadm, gen, eth, diag, list(geohe)]
                except:
                    pass
            else:
                continue

    if ge == 0:
        return dict(male_info, **female_info), merged_df
    elif ge == 1:
        return male_info, merged_df
    else:
        return female_info, merged_df
    # return total_info_list

# total = get_patients_gender()
# total = dict(male, **female)
# print('test')
