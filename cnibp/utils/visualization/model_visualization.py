import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.cuda as cuda
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from cnibp.dataset_loader import dataset_loader
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from cnibp.utils.visualization.visualization import graph_plot as gp

# mixer.init()
# sound = mixer.Sound('bell-ringing-01c.wav')

''' warning: do not turn on set_detect_anomaly(True) when training, only for debugging '''
# torch.autograd.set_detect_anomaly(True)

with open('/home/paperc/PycharmProjects/Pytorch_rppgs/cnibp/configs/parameter.json') as f:
    json_data = json.load(f)
    # param = json_data.get("parameters")
    channels = json_data.get("parameters").get("in_channels")
    gender = json_data.get("parameters").get("gender")
    # out_channels = json_data.get("parameters").get("out_channels")
    # hyper_param = json_data.get("hyper_parameters")
    # wb = json_data.get("wandb")
    root_path = json_data.get("parameters").get("root_path")  # mimic uci containing folder
    data_path = json_data.get("parameters").get("dataset_path")  # raw mimic uci selection
    # sampling_rate = json_data.get("parameters").get("sampling_rate")
    models = json_data.get("parameters").get("models")
    # cases = json_data.get("parameters").get("cases")

print(sys.version)

# GPU Setting
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('----- GPU INFO -----\nDevice:', DEVICE)
print('Count of using GPUs:', torch.cuda.device_count())
print('Current cuda device:', torch.cuda.current_device())
gpu_ids = list(map(str, list(range(torch.cuda.device_count()))))
total_gpu_memory = 0
for gpu_id in gpu_ids:
    total_gpu_memory += torch.cuda.get_device_properties("cuda:" + gpu_id).total_memory
print('Total GPU Memory :', total_gpu_memory, '\n--------------------')

if torch.cuda.is_available():
    random_seed = 125
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    cuda.manual_seed(random_seed)
    cuda.allow_tf32 = True
    # torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    cudnn.enabled = True
    cudnn.deterministic = True  # turn on for reproducibility ( if turned on, slow down training )
    cudnn.benchmark = False
    cudnn.allow_tf32 = True
else:
    print("cuda not available")

channel_info = channels['P']
gender_info = gender['Total']
"""model setup"""
# model, loss, optimizer, scheduler = get_model('BPNet', channel=channel_info[0], device=DEVICE, stage=1)
# model_2, loss_2, optimizer_2, scheduler_2 = get_model(model_name, device=DEVICE, stage=2)
epochs = 1
"""dataset setup"""
dataset = dataset_loader(dataset_name='mimiciii', in_channel=channel_info[-1], batch_size=1024,
                         device=DEVICE,
                         gender=gender_info[-1])


def dbp_sbp_scatter(test_dataset):
    '''
    Visualize model prediction (PPG -> DBP/SBP)
    '''
    model = torch.load(
        '/home/paperc/PycharmProjects/Pytorch_rppgs/cnibp/weights/BPNet_mimiciii_20230207_125656_P+V+A_Total.pt')
    model.to(DEVICE)
    model.eval()
    x = np.array(range(40, 200))
    for e in range(epochs):
        with tqdm(test_dataset, desc='Test-{}'.format(str(e)), total=len(test_dataset), leave=True) as test_epoch:
            with torch.no_grad():
                for idx, (X_test, Y_test, d, s, m, info, ohe) in enumerate(test_epoch):
                    hypothesis, dbp, sbp, mbp = model(X_test)
                    plt.figure(figsize=(10, 10))
                    plt.xlim(40, 200)
                    plt.ylim(40, 200)
                    plt.scatter(d.detach().cpu(), torch.squeeze(dbp).detach().cpu(), c='blue', marker='x', alpha=0.1)
                    plt.scatter(s.detach().cpu(), torch.squeeze(sbp).detach().cpu(), c='red', alpha=0.1)
                    plt.plot(x, x, color='k', label='y=x')
                    plt.grid(color='gray', alpha=.5, linestyle='--')
                    plt.xlabel('Target BP')
                    plt.ylabel('Predicted BP')
                    plt.legend()
                plt.show()
    # np.random.seed(0)

    # print('test')


# dbp_sbp_scatter(dataset[-1])

def get_cluster_n(X):
    sse = []
    for i in range(1, 15):
        km = KMeans(n_clusters=i, init='k-means++', random_state=0)
        km.fit(X)
        sse.append(km.inertia_)
    plt.plot(range(1, 15), sse, marker='o')
    plt.title('Elbow')
    plt.xlabel('Cluster N')
    plt.ylabel('SSE')
    plt.show()
    plt.close()
    silhouette_val = []
    for i in range(2, 15):
        km_sil = KMeans(n_clusters=i, init='k-means++', random_state=0)
        pred = km_sil.fit_predict(X)
        silhouette_val.append(np.mean(silhouette_samples(X, pred, metric='euclidean')))
    plt.plot(range(2, 15), silhouette_val, marker='o')
    plt.title('Silhouette')
    plt.xlabel('Cluster N')
    plt.ylabel('Silhouette')
    plt.show()
    return 10


def cluster_by_abp_shape(in_dataset, is_abp, test_n=100, check_point_n=10):
    # mode = ['Train', 'Validation', 'Test']
    total_cnt = 0
    scaler = MinMaxScaler()
    resample_len = 100
    resampled_t = np.arange(0, resample_len, 1)
    check_point = [x for x in resampled_t if x % check_point_n == 0]
    np.random.seed(125)

    original_abp_cycles = []
    abp_cycles = []
    shape_info = []
    patient_info = []
    patient_id = []
    for i in range(3):
        for ppg, ppg_cycle, abp, abp_cycle, d, s, info in tqdm(in_dataset[i]):
            mask = np.random.choice(np.arange(len(abp)), test_n)
            selected_ppg = ppg[mask].detach().cpu().numpy()
            selected_abp = abp[mask].detach().cpu().numpy()
            selected_abp_cycle = abp_cycle[mask].detach().cpu().numpy()
            selected_info = info[mask].detach().cpu().numpy()
            for sa, sac, si in zip(selected_abp, selected_abp_cycle, selected_info):
                ''' label 11 means Non-Cardiac Patients'''
                if int(si[-1]) == 11:
                    continue
                original_abp_cycles.append(sac)
                normalized_cycle = np.squeeze(scaler.fit_transform(np.reshape(sac, (-1, 1))))
                abp_cycles.append(normalized_cycle)
                shape_info.append(normalized_cycle[check_point])
                patient_info.append(int(si[-1]))
                patient_id.append(int(si[0]))
                total_cnt += 1

    shape_info = np.array(shape_info)
    cluster_n = get_cluster_n(shape_info)
    kmeans = KMeans(n_clusters=cluster_n, init='k-means++', random_state=0).fit(shape_info)

    plt.figure(figsize=(10, 10))
    plt.xlim(0, 13)
    plt.ylim(0, cluster_n)
    plt.xlabel('Disease Labels')
    plt.ylabel('Cluster Labels')
    for i in range(cluster_n):
        cluster = np.array(patient_info)[list(np.where(kmeans.labels_ == i))]
        disease_info = np.unique(cluster, return_counts=True)
        disease = disease_info[0]
        scatter_size = disease_info[-1]
        plt.scatter(disease, np.zeros_like(disease) + i, s=scatter_size)
    plt.show()
    plt.close()

    for i in range(10):
        disease_label, disease_cnt = np.unique(np.array(patient_info)[list(np.where(kmeans.labels_ == i))],
                                               return_counts=True)
        gp.plot_pie_chart('cluster {}'.format(str(i)), disease_label, disease_cnt)
        print('cluster', i, 'disease_label :', disease_label, 'disease_cnt :', disease_cnt,
              '\nhighest_portion :', np.max(disease_cnt/np.sum(disease_cnt)),
              'highest_cnt cluster :', disease_label[np.argmax(disease_cnt)])

    for i in range(cluster_n):
        plt.plot(kmeans.cluster_centers_[i], label='Cluster {}'.format(str(i)))
        # plt.plot(kmeans.cluster_centers_[1], label='Cluster {}'.format(str('Abnormal')))
    plt.title('ABP Cycle Clustering')
    plt.xlabel('Resampled ABP length (nu)')
    plt.ylabel('Normalized ABP Amplitude (nu)')
    plt.legend()
    plt.show()
    plt.close()
    '''BPNet Figure - single cycle '''
    sample_time = np.arange(0, 1, 1 / 60)[:len(np.array(original_abp_cycles)[list(np.where(kmeans.labels_ == 0))][16])]
    plt.plot(sample_time, np.array(original_abp_cycles)[list(np.where(kmeans.labels_ == 0))][16], 'r')
    plt.axhline(y=np.max(np.array(original_abp_cycles)[list(np.where(kmeans.labels_ == 0))][16]), color='lightgrey',
                linestyle='--', label='Systolic Blood Pressure')
    # plt.axhline(y=74, color='lightgrey', linestyle='-.', label='Dicrotic Notch')
    plt.axhline(y=np.min(np.array(original_abp_cycles)[list(np.where(kmeans.labels_ == 0))][16]), color='lightgrey',
                linestyle='-.', label='Diastolic Blood Pressure')
    plt.annotate('Dicrotic Notch', xy=(0.016666667 * 23, 76),
                 xytext=(0.016666667 * 23, 74 + 10),
                 arrowprops=dict(arrowstyle='-|>', connectionstyle='angle, angleA=180, angleB=270, rad=5',
                                 color='gray'))
    plt.xlabel('Time (s)')
    plt.ylabel('Arterial Blood Pressure (mmHg)')
    plt.title('Single Cycle of Arterial Blood Pressure')
    plt.legend(loc='upper right')
    plt.show()
    print('total_cnt :', total_cnt)
    pass


cluster_by_abp_shape(dataset, True, test_n=1000, check_point_n=1)


def dbp_sbp_distribution_checker(in_dataset):
    '''BPNet Figure - SBP, DBP Distribution'''
    mode = ['Train', 'Validation', 'Test']
    disease_cnt = np.zeros(15)
    # wedgeprops = {'width': 0.7, 'edgecolor': 'w', 'linewidth': 5}
    # explode = [0.05, 0.10, 0.05, 0.05, 0.05, 0.1, 0.05, 0.05, 0.05, 0.1, 0.05, 0.05, 0.05, 0.1]
    # colors = ['#ff9999', '#ffc000', '#8fd9b6', '#d395d0', '#ff9999', 'silver', 'gold', 'lightcyan', 'lightgrey',
    #           'dodgerblue', 'violet', '#e35f62', 'seagreen']
    cnt = 0
    wave0, wave1, wave2, wave3, wave4, wave5, wave6, wave7, wave8, wave9, wave10, wave11, wave12, wave13, wave14 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    disease_waveform = [wave0, wave1, wave2, wave3, wave4, wave5, wave6, wave7, wave8, wave9, wave10, wave11,
                        wave12, wave13, wave14]
    disease_sbp = []
    disease_dbp = []
    for i in range(3):
        test_dbp_l = np.zeros((300,), dtype=int)
        test_sbp_l = np.zeros((300,), dtype=int)
        # bp = np.arange(300)
        subjects_id = set()
        # disease_cnt = np.zeros(15)
        # subjects_disease = set()
        disease_name = ['ANGINA', 'AORTIC DISSECTION', 'AORTIC STENOSIS', 'CARDIAC ARREST', 'CARDIAC INFARCTION',
                        'CARDIOMYOPATHY', 'CHEST PAIN', 'CORONARY ARTERY DISEASE', 'HEART FAILURE', 'HYPERTENSION',
                        'HYPOTENSION', 'NON CARDIAC DISEASE', 'STROKE', 'VALVULAR DISEASE', 'VENTRICULAR \nTACHYCARDIA']
        # wave = []

        data = in_dataset[i]
        for e in range(epochs):
            with tqdm(data, desc='Dataset-{}'.format(str(i)), total=len(data), leave=True) as dset:
                with torch.no_grad():
                    for idx, (X_test, Y_test, d, s, m, info, ohe) in enumerate(dset):
                        # plotting distribution
                        # abp_wave = np.array(info.detach().cpu()[:-1], dtype=int)
                        # wave.append(list(Y_test[np.where(abp_wave == 0)].detach().cpu().numpy()))
                        for j in range(15):
                            w = list(Y_test[np.where(
                                np.array(info.detach().cpu()[:, -1], dtype=int) == j)].detach().cpu().numpy())
                            if len(w) != 0:
                                if cnt == 0:
                                    disease_waveform[j].append(w)
                                else:
                                    np.append(disease_waveform[j], np.vstack(w))
                            else:
                                continue
                        for abp, dbp, sbp, inf in zip(Y_test, d, s, info):
                            test_dbp_l[int((torch.round(dbp)).detach().cpu())] += 1
                            test_sbp_l[int((torch.round(sbp)).detach().cpu())] += 1
                            subjects_id.add(int(inf.detach().cpu().numpy()[0]))
                            disease_cnt[int(inf[-1])] += 1
                            # subjects_disease.add(int(inf.detach().cpu().numpy()[-1]))
        for k in range(15):
            try:
                disease_waveform[k] = np.vstack(disease_waveform[k])
            except:
                pass
        for l in range(15):
            try:
                if cnt == 0:
                    disease_sbp.append(np.max(disease_waveform[l], axis=-1))
                    disease_dbp.append(np.min(disease_waveform[l], axis=-1))
                else:
                    np.append(disease_sbp[l], np.max(disease_waveform[l], axis=-1))
                    np.append(disease_dbp[l], np.min(disease_waveform[l], axis=-1))
            except:
                pass
        fig, ax = plt.subplots()
        ax.boxplot(disease_sbp, notch=True, whis=2.5, vert=True)
        plt.show()
        plt.close()
        fig, ax = plt.subplots()
        ax.boxplot(disease_dbp, notch=True, whis=2.5, vert=True)
        plt.show()
        '''BPNet Figure Pie chart'''
        gp.plot_pie_chart('Cumulative CVD Ratio of Datasets',
                          list(np.delete(disease_cnt, 11)), np.delete(disease_name, 11), False, '%1.1f%%', True)
        # plt.figure(figsize=(10, 8))
        # plt.title('Cumulative CVD Ratio of Datasets')
        # plt.pie(list(np.delete(disease_cnt, 11)), labels=np.delete(disease_name, 11), autopct='%.1f%%',
        #         colors=colors, counterclock=False, explode=explode, shadow=True)  # , wedgeprops=wedgeprops)
        # plt.show()
        # plt.close()
        plt.figure(figsize=(10, 8))
        plt.title(mode[cnt] + ' Dataset Distribution' + ' ({} Subjects)\n'.format(str(len(subjects_id))))
        plt.xlabel('Arterial Blood Pressure (mmHg)')
        plt.ylabel('Segments (n)')
        dbp_min = np.where(test_dbp_l != 0)[0][0]
        dbp_max = np.where(test_dbp_l != 0)[0][-1]
        sbp_min = np.where(test_sbp_l != 0)[0][0]
        sbp_max = np.where(test_sbp_l != 0)[0][-1]
        dbp_len = np.sum(test_dbp_l)
        sbp_len = np.sum(test_sbp_l)
        # dbp_cumsum = np.cumsum(test_dbp_l)
        # dbp_cumsum_reversed = np.cumsum(np.flip(test_dbp_l))
        # sbp_cumsum = np.cumsum(test_sbp_l)
        # sbp_cumsum_reversed = np.cumsum(np.flip(test_sbp_l))
        dbp_10_line = int(np.where(np.cumsum(test_dbp_l) > int(dbp_len * 0.1))[0][0])
        dbp_90_line = 300 - int(np.where(np.cumsum(np.flip(test_dbp_l)) > int(dbp_len * 0.1))[0][0])
        sbp_10_line = int(np.where(np.cumsum(test_sbp_l) > int(sbp_len * 0.1))[0][0])
        sbp_90_line = 300 - int(np.where(np.cumsum(np.flip(test_sbp_l)) > int(sbp_len * 0.1))[0][0])
        # plt.bar(bp, test_dbp_l, width=1.0, color='#0066ff', label='Diastolic({} segments)'.format(np.sum(test_dbp_l)))
        # plt.bar(bp, test_sbp_l, width=1.0, color='#ff5050', label='Systolic({} segments)'.format(np.sum(test_sbp_l)))
        plt.vlines(x=dbp_10_line, color='k', linestyle='dotted', ymin=0, ymax=test_dbp_l[dbp_10_line])
        plt.vlines(x=dbp_90_line, color='k', linestyle='dotted', ymin=0, ymax=test_dbp_l[dbp_90_line])
        plt.vlines(x=sbp_10_line, color='k', linestyle='dotted', ymin=0, ymax=test_sbp_l[sbp_10_line])
        plt.vlines(x=sbp_90_line, color='k', linestyle='dotted', ymin=0, ymax=test_sbp_l[sbp_90_line])
        plt.plot(test_dbp_l, color='#0066ff', linestyle='solid', linewidth=2,
                 label='Diastolic({} segments)'.format(np.sum(test_dbp_l)))
        plt.plot(test_sbp_l, color='#ff5050', linestyle='solid', linewidth=2,
                 label='Systolic({} segments)'.format(np.sum(test_sbp_l)))
        plt.annotate('10% of DBP', xy=(dbp_10_line, test_dbp_l[dbp_10_line] // 2),
                     xytext=(dbp_10_line - 50, test_dbp_l[dbp_10_line] - 5),
                     arrowprops=dict(arrowstyle='-|>', connectionstyle='angle,angleA=-90, angleB=180, rad=5',
                                     color='gray'))
        plt.annotate('90% of DBP', xy=(dbp_90_line, test_dbp_l[dbp_90_line] // 2),
                     xytext=(dbp_90_line, test_dbp_l[dbp_10_line]),
                     arrowprops=dict(arrowstyle='-|>', connectionstyle='angle,angleA=-90, angleB=180, rad=5',
                                     color='gray'))
        plt.annotate('10% of SBP', xy=(sbp_10_line, test_sbp_l[sbp_10_line] // 2),
                     xytext=(sbp_10_line + 10, test_sbp_l[sbp_10_line] - 50),
                     arrowprops=dict(arrowstyle='-|>', connectionstyle='angle,angleA=-90, angleB=180, rad=5',
                                     color='gray'))
        plt.annotate('90% of SBP', xy=(sbp_90_line, test_sbp_l[sbp_90_line] // 2),
                     xytext=(sbp_90_line + 10, test_sbp_l[sbp_10_line] - 50),
                     arrowprops=dict(arrowstyle='-|>', connectionstyle='angle,angleA=-90, angleB=180, rad=5',
                                     color='gray'))

        if i == 0:
            plt.annotate('Min val of DBP ({})'.format(dbp_min), xy=(dbp_min, test_dbp_l[dbp_min]),
                         xytext=(dbp_min - 25, 200), arrowprops=dict(arrowstyle='->', color='gray'))
            plt.annotate('Max val of DBP ({})'.format(dbp_max), xy=(dbp_max, test_dbp_l[dbp_max]),
                         xytext=(dbp_max - 25, 200), arrowprops=dict(arrowstyle='->', color='gray'))
            plt.annotate('Min val of SBP ({})'.format(sbp_min), xy=(sbp_min, test_sbp_l[sbp_min]),
                         xytext=(sbp_min - 20, 500), arrowprops=dict(arrowstyle='->', color='gray'))
            plt.annotate('Max val of SBP ({})'.format(sbp_max), xy=(sbp_max, test_sbp_l[sbp_max]),
                         xytext=(sbp_max - 20, 500), arrowprops=dict(arrowstyle='->', color='gray'))
        else:
            plt.annotate('Min val of DBP ({})'.format(dbp_min), xy=(dbp_min, test_dbp_l[dbp_min]),
                         xytext=(dbp_min - 25, test_dbp_l[dbp_min] + 20),
                         arrowprops=dict(arrowstyle='->', color='gray'))
            plt.annotate('Max val of DBP ({})'.format(dbp_max), xy=(dbp_max, test_dbp_l[dbp_max]),
                         xytext=(dbp_max - 25, test_dbp_l[dbp_max] + 20),
                         arrowprops=dict(arrowstyle='->', color='gray'))
            plt.annotate('Min val of SBP ({})'.format(sbp_min), xy=(sbp_min, test_sbp_l[sbp_min]),
                         xytext=(sbp_min - 20, test_sbp_l[sbp_min] + 50),
                         arrowprops=dict(arrowstyle='->', color='gray'))
            plt.annotate('Max val of SBP ({})'.format(sbp_max), xy=(sbp_max, test_sbp_l[sbp_max]),
                         xytext=(sbp_max - 20, test_sbp_l[sbp_max] + 50),
                         arrowprops=dict(arrowstyle='->', color='gray'))
        plt.legend()
        plt.show()
        cnt += 1


# dbp_sbp_distribution_checker(dataset)


def dbp_sbp_correlation_checker(in_dataset):
    mode = ['Train', 'Validation', 'Test']
    for i in range(3):
        test_dbp_l = np.zeros((300,), dtype=int)
        test_sbp_l = np.zeros((300,), dtype=int)
        dbp_list = np.zeros((10000,), dtype=float)
        sbp_list = np.zeros((10000,), dtype=float)
        # bp = np.arange(300)
        subjects = set()
        data = in_dataset[i]
        ii = 0
        plt.figure(figsize=(10, 10))
        plt.xlim(0, 240)
        plt.ylim(0, 240)
        for e in range(epochs):
            with tqdm(data, desc='Dataset-{}'.format(str(i)), total=len(data), leave=True) as dset:
                with torch.no_grad():
                    for idx, (X_test, Y_test, d, s, m, info, ohe) in enumerate(dset):
                        # print(X_test)
                        plt.scatter(d.detach().cpu(), s.detach().cpu(), color='k', alpha=0.1)
                        # for dbp, sbp, inf in zip(d, s, info):
                        #     if ii > 9999:
                        #         break
                        #     dbp_list[ii] = dbp
                        #     sbp_list[ii] = sbp
                        #     ii += 1

        # plt.close()

        # plt.figure(figsize=(10, 10))
        # plt.title(mode[i] + ' Dataset Distribution' + ' ({} Subjects)\n'.format(str(len(subjects))))
        t = np.arange(240)
        ty = t * 3. - 50
        plt.plot(t, ty)
        plt.xlabel('Diastolic Blood Pressure (mmHg)')
        plt.ylabel('Systolic Blood Pressure (mmHg)')

        plt.show()

# dbp_sbp_correlation_checker(dataset)
