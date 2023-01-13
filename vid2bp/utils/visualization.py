from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from vid2bp.dataset_loader import dataset_loader
import numpy as np
import random
from tqdm import tqdm
from matplotlib import cm
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import vid2bp.preprocessing.utils.signal_utils as su
from scipy.signal import resample
from heartpy.filtering import filter_signal
from PyEMD import EMD, EEMD, Visualisation

dataset_name = 'mimiciii'
channel = 3
batch_size = 512
dataset = dataset_loader(dataset_name=dataset_name, channel=1, batch_size=1)

def emd(dataset):
    train_dataset = dataset[0]

    for ple, _, _, abp, _, _, _, _ in train_dataset:
        ple_sig = ple.view(360, ).detach().cpu().numpy()
        abp_sig = abp.view(360, ).detach().cpu().numpy()
        emd = EMD()

        emd.emd(ple_sig, max_imf=2)

        imfs, res = emd.get_imfs_and_residue()

        vis = Visualisation()
        vis.plot_imfs(imfs=imfs, residue=res, t=np.arange(0, 6, 1 / 60), include_residue=True)
        plt.plot(np.arange(0, 6, 1 / 60), ple_sig, c='k', alpha=0.5, linewidth=0.5)
        # vis.plot_instant_freq(np.arange(0, 6, 1 / 60), imfs=imfs)
        vis.show()
        plt.show()
        pass

emd(dataset)
def get3dplot(dataset):
    train_dataset = dataset[0]
    plt.rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.2)
    fig = plt.figure(figsize=(8, 8))
    # fig.suptitle('\n\n\n\nPhotoplethysmography & Arterial Blood Pressure', fontsize=14)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('PPG & ABP Projection', fontsize=14)
    ax.set_xlim3d(0, 6)
    ax.set_ylim3d(220, 0)
    ax.set_zlim3d(0, 4)
    ax.set_xlabel('time (t)')
    ax.set_ylabel('ABP (mmHg)')
    ax.set_zlabel('PPG')
    t_real = np.arange(0, 6, 1 / 60)
    t = np.zeros(360)
    # cmin, cmax = 0, 2
    cnt = 0
    low_cnt = 0
    high_cnt = 0
    # total_cnt = low_cnt + high_cnt
    for ple, abp, d, s in train_dataset:
        cnt += 1
        # color = np.array([(cmax - cmin) * np.random.random_sample() + cmin for i in range(360)])
        ple_sig = ple.view(360, ).detach().cpu().numpy()
        abp_sig = abp.view(360, ).detach().cpu().numpy()
        if np.mean(ple_sig) < 1.:
            low_cnt += 1
            ax.plot(t_real, abp_sig, zs=0, zdir='z', c='k', alpha=0.5, linewidth=0.5)
            ax.plot(t_real, zs=0, ys=ple_sig, zdir='y', c='k', alpha=0.5, linewidth=0.5)
            ax.scatter(t, abp_sig, ple_sig, c='b', marker='.', alpha=random.randint(25, 50) * 0.02, s=2)
            ax.scatter(t_real, abp_sig, ple_sig, c='b', marker='.', alpha=random.randint(25, 50) * 0.02, s=3)
        else:
            if high_cnt > 0:
                continue
            else:
                high_cnt += 1
                ax.plot(t_real, abp_sig, zs=0, zdir='z', c='k', alpha=0.5, linewidth=0.5)
                ax.plot(t_real, zs=0, ys=ple_sig, zdir='y', c='k', alpha=0.5, linewidth=0.5)
                ax.scatter(t, abp_sig, ple_sig, c='r', marker='.', alpha=random.randint(25, 50) * 0.02, s=2)
                ax.scatter(t_real, abp_sig, ple_sig, c='r', marker='.', alpha=random.randint(25, 50) * 0.02, s=3)

        if low_cnt + high_cnt > 1 and low_cnt > 0:
            break
    # ax.view_init(elev=20., azim=-35)
    plt.show()


# get3dplot(dataset)

def get_scale_shape_plot(dataset, n=len(dataset[-1])):
    test_dataset = dataset[-1]
    valid_dataset = dataset[1]
    n = len(test_dataset)
    plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111)
    cnt = 0
    ple_amp = []
    abp_amp = []
    for ple, abp, d, s in tqdm(test_dataset):
        cnt += 1
        ple_sig = ple.view(360, ).detach().cpu().numpy()
        ple_amp.append(np.max(ple_sig) - np.min(ple_sig))
        abp_amp.append(((2 * d + s) / 3).detach().cpu().numpy())

    for ple, abp, d, s in tqdm(valid_dataset):
        cnt += 1
        ple_sig = ple.view(360, ).detach().cpu().numpy()
        ple_amp.append(np.max(ple_sig) - np.min(ple_sig))
        abp_amp.append(((2 * d + s) / 3).detach().cpu().numpy())

    abp_amp = np.squeeze(np.array(abp_amp)).tolist()
    abp_size = (abp_amp / np.max(abp_amp)).tolist()
    plt.scatter(ple_amp, abp_amp,
                # s=1.0,
                c=abp_amp, marker='.', alpha=0.5, cmap=cm.jet, vmin=50, vmax=180, s=abp_size)
    plt.xlabel('PLE amplitude', fontsize=12)
    plt.ylabel('Mean Arterial Pressure (mmHg)', fontsize=12)
    plt.title('PLE amplitude & MAP', fontsize=14)
    plt.colorbar()
    plt.show()


# get_scale_shape_plot(dataset)

def ple_clustering(dataset, test_n=100, cluster_num=5, check_point_n=10, plot_shape_flag=False,
                   plot_class_map_flag=True, plot_3d_flag=True):
    scaler = MinMaxScaler()
    test_dataset = dataset[0]
    cnt = 0
    plot_cnt = 0
    abp_shape_info = []
    shape_info = []  # normalize(0~1) & resampled(100) ple의 check_point의 값( ple의 shape 정보 )
    diff_info = []
    check_points = []
    # class_mixed_info = []  # class_info(ple shape 정보) + ple_amp(ple 크기 정보) + ple_len(ple 주파수 정보)
    original_cycle = []  # plot 찍을 때 사용할 normalized(0~1) & resampled ple
    original_abp_cycle = []
    original_map = []  # plot 찍을때 사용할 ple에 매칭되는 abp의 map
    original_sbp = []  # plot 찍을때 사용할 ple에 매칭되는 abp의 sbp
    original_amp = []  # clustering 할때 사용할 해당 ple의 크기
    original_len = []  # clustering 할때 사용할 해당 ple의 주파수
    # t = np.arange(0, 6, 1 / 60)
    resample_len = 100
    resampled_t = np.arange(0, resample_len, 1)
    for ple, abp, d, s in tqdm(test_dataset):
        ple_sig = (ple.view(360, 1)).detach().cpu().numpy()
        abp_sig = (abp.view(360, 1)).detach().cpu().numpy()
        hf = 8
        denoised_ple = filter_signal(np.squeeze(ple_sig), cutoff=hf, sample_rate=60, order=2, filtertype='lowpass')
        denoised_abp = filter_signal(np.squeeze(abp_sig), cutoff=hf, sample_rate=60, order=2, filtertype='lowpass')
        # dbp_index = su.DBP_detection(denoised_ple)
        abp_low_index = su.DBP_detection(denoised_abp)
        dbp_index = su.DBP_detection(np.squeeze(denoised_ple))
        # abp_low_index = su.DBP_detection(np.squeeze(denoised_abp))
        try:
            temp = ple_sig[dbp_index[0]:dbp_index[1]]
            single_cycle = np.squeeze(scaler.fit_transform(ple_sig[dbp_index[0]:dbp_index[1]]))
            # single_cycle = np.squeeze((ple_sig[dbp_index[0]:dbp_index[1]]))
            raw_abp_cycle = np.squeeze(abp_sig[abp_low_index[0]:abp_low_index[1]])
            abp_single_cycle = np.squeeze(scaler.fit_transform(abp_sig[abp_low_index[0]:abp_low_index[1]]))
            resampled_cycle = resample(single_cycle, resample_len)
            # apg = resample(np.diff(np.diff(single_cycle)), resample_len)
            abp_resampled_cycle = resample(abp_single_cycle, resample_len)
            check_point = [x for x in resampled_t if x % check_point_n == 0]
            # if np.where(resampled_cycle == np.max(resampled_cycle))[0] > 50:
            #     plot_cnt+=1
            #     plt.plot(np.squeeze(scaler.fit_transform(ple_sig[:200])), 'g', label='ple')
            #     plt.plot(np.squeeze(scaler.fit_transform(abp_sig[:200])), 'r', label='abp')
            #     plt.plot(np.squeeze(scaler.fit_transform(np.max(ple_sig[:200]) - ple_sig[:200])), 'b', label='reversed ple')
                # if plot_cnt==1:
                #     plt.legend()
                #     plt.show()
                    # return
        except:
            continue

        if abs(resampled_cycle[0] - resampled_cycle[-1]) > 0.05:
            continue
        else:
            check_points.append(check_point)
            original_cycle.append(resampled_cycle)
            original_abp_cycle.append(abp_resampled_cycle)
            original_len.append(len(single_cycle))
            # if original_len[-1] <30:
            #     plt.plot(temp)
            original_amp.append(np.max(temp) - np.min(temp))
            original_map.append(((2 * d + s) / 3).detach().cpu().numpy())
            original_sbp.append(s.detach().cpu().numpy())
            shape_info.append(resampled_cycle[check_point])
            abp_shape_info.append(abp_resampled_cycle[check_point])
            # diff_info.append(apg[check_point])
            cnt += 1

        if cnt == test_n:
            break
    plt.show()
    shape_info = np.array(shape_info)
    abp_shape_info = np.array(abp_shape_info)
    freq_info = scaler.fit_transform(np.reshape(original_len, (-1, 1)))  # ple_freq
    freq_info2 = [60 / (x / 60) for x in np.reshape(original_len, (test_n, 1))]
    freq_info2 = np.reshape([x if x < 200 else 200 for x in freq_info2], (-1, 1))
    shape_freq_info = np.concatenate((shape_info, freq_info), axis=-1)  # ple_shape + ple_freq
    # amp_info = scaler.fit_transform(np.reshape(original_amp, (-1, 1)))  # ple_amp
    amp_info = np.reshape(original_amp, (test_n, 1))  # ple_amp
    shape_amp_info = np.concatenate((shape_info, amp_info), axis=-1)  # ple_shape + ple_amp
    shape_freq_amp_info = np.concatenate((shape_freq_info, amp_info), axis=-1)  # ple_shape + ple_freq + ple_amp
    kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(shape_info)
    # kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(np.array(diff_info))
    # print(kmeans.cluster_centers_)
    plt.figure(figsize=(8, 8))
    plt.title('Clustered PLE shape (cluster num = {})'.format(cluster_num), fontsize=14)
    plt.xlabel('time (resampled)', fontsize=12)
    plt.ylabel('PLE amplitude (normalized, 0~1)', fontsize=12)
    for i in range(cluster_num):
        plt.plot(kmeans.cluster_centers_[i], label='cluster {}'.format(i))
    plt.legend()
    plt.show()
    plt.close()
    '''@@@@@@'''
    # for i in range(cluster_num):
    #     plt.figure(figsize=(8, 8))
    #
    #     cluster_cnt = 0
    #     sbp_sum = 0
    #     sbp_list = []
    #     for j in range(test_n):
    #         if kmeans.labels_[j] == i:
    #             cluster_cnt += 1
    #             plt.plot(original_cycle[j], label='cluster {}'.format(i))
    #     plt.title('Clustered PLE shape (cluster {}, n={})'.format(i, cluster_cnt), fontsize=14)
    #     plt.xlabel('time (resampled)', fontsize=12)
    #     plt.ylabel('PLE amplitude (normalized, 0~1)', fontsize=12)
    #     # plt.legend()
    #     if cluster_cnt > 10:
    #         plt.show()
    #     plt.close()
    #
    #     # ABP
    #     plt.figure(figsize=(8, 8))
    #     for j in range(test_n):
    #         if kmeans.labels_[j] == i:
    #             sbp_list.append(original_sbp[j])
    #             plt.plot(original_abp_cycle[j], label='cluster {}'.format(i))
    #     # sbp_mean = round(float(np.mean(sbp_list)), 2)
    #     plt.title('Clustered ABP shape (cluster {}, n={})\nsbp mean={}, sbp std={}'.format(i, cluster_cnt,
    #                                                                                        round(float(np.mean(sbp_list)),2),
    #                                                                                        round(float(np.std(sbp_list)),2)), fontsize=14)
    #     plt.xlabel('time (resampled)', fontsize=12)
    #     plt.ylabel('ABP amplitude (normalized, 0~1)', fontsize=12)
    #     # plt.legend()
    #     if cluster_cnt > 10:
    #         plt.show()
    #     plt.close()
    #     # ppg abp shape not matched
    #     # plt.figure(figsize=(8, 8))
    #     # for j in range(test_n):
    #     #     if kmeans.labels_[j] == i:
    #     #         if original_abp_cycle

    abp_kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(abp_shape_info)
    for i in range(cluster_num):
        if np.argmax(abp_kmeans.cluster_centers_[i])>49:
            plt.plot(abp_kmeans.cluster_centers_[i], label='cluster {}'.format(i))
            plt.legend()
            plt.show()
    # print(abp_kmeans.cluster_centers_)
    plt.figure(figsize=(8, 8))
    plt.title('Clustered ABP shape (cluster num = {})'.format(cluster_num), fontsize=14)
    plt.xlabel('time (resampled)', fontsize=12)
    plt.ylabel('ABP amplitude (normalized, 0~1)', fontsize=12)
    for i in range(cluster_num):
        plt.plot(abp_kmeans.cluster_centers_[i], label='cluster {}'.format(i))
    plt.legend()
    plt.show()
    plt.close()

    kmeans2 = KMeans(n_clusters=cluster_num, random_state=0).fit(freq_info2)
    print(kmeans2.cluster_centers_)
    plt.figure(figsize=(8, 8))
    plt.title('Clustered PLE freq (cluster num = {})'.format(cluster_num), fontsize=14)
    plt.xlabel('cluster', fontsize=12)
    plt.ylabel('PLE freq (bpm)', fontsize=12)
    for i in range(cluster_num):
        plt.bar(i, height=kmeans2.cluster_centers_[i], label='cluster {}'.format(i))
    plt.legend()
    plt.show()
    plt.close()
    # kmeans3 = KMeans(n_clusters=cluster_num, random_state=0).fit(freq_info)

    print('plotting ppg freq info graph... ')
    plt.figure(figsize=(8, 8))
    plt.xlabel('Heart Rate (BPM)', fontsize=12)
    plt.ylabel('SBP (mmHg)', fontsize=12)
    plt.xlim(40, 200)
    plt.ylim(50, 200)
    plt.title('Scatter plot with Heart Rate & SBP', fontsize=14)
    for i in tqdm(range(len(original_sbp))):
        if freq_info2[i] > 200:
            continue
        else:
            plt.scatter(freq_info2[i], original_sbp[i], c=original_sbp[i], marker='.', alpha=0.5, cmap=cm.jet, vmin=50,
                        vmax=200)
    plt.colorbar()
    plt.show()
    plt.close()

    print('plotting ppg amp info graph... ')
    plt.figure(figsize=(8, 8))
    plt.xlabel('PPG Amplitude', fontsize=12)
    plt.ylabel('SBP (mmHg)', fontsize=12)
    plt.xlim(0, 2.5)
    plt.ylim(50, 200)
    plt.title('Scatter Plot with PPG Amplitude & SBP', fontsize=14)
    for i in tqdm(range(len(original_sbp))):
        if freq_info2[i] > 200:
            continue
        else:
            plt.scatter(amp_info[i], original_sbp[i], c=original_sbp[i], marker='.', alpha=0.5, cmap=cm.jet, vmin=50,
                        vmax=200)
    plt.colorbar()
    plt.show()
    plt.close()

    if plot_3d_flag:
        plt.rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.2)
        original_sbp_sorted = np.sort(np.squeeze(original_sbp))
        original_sbp_sorted_index = np.argsort(np.squeeze(original_sbp))

        labels = kmeans.labels_[original_sbp_sorted_index]
        labels2 = kmeans2.labels_[original_sbp_sorted_index]
        label_sorted_by_sbp = [labels[x] for x in original_sbp_sorted_index]
        label2_sorted_by_sbp = [labels2[x] for x in original_sbp_sorted_index]
        fig = plt.figure(figsize=(8, 8))
        plt.title('Clustered by PPG Shape and Frequency & SBP', fontsize=14)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim3d(cluster_num, 0)
        ax.set_ylim3d(0, cluster_num)
        ax.set_zlim3d(50, 200)
        ax.set_xlabel('Clustered by Plethysmo Shape')
        ax.set_ylabel('Clustered by Plethysmo Amplitude')
        ax.set_zlabel('SBP (mmHg)')
        # t_class = np.arange(0, cluster_num, 1)
        print('plotting 3d graph... ')
        for i in tqdm(range(len(original_cycle))):
            try:
                ax.scatter(label_sorted_by_sbp[i], label2_sorted_by_sbp[i], original_sbp_sorted[i],
                           color=cm.jet((original_sbp_sorted[i] / np.max(original_sbp_sorted)) - 0.05))
                # ax.scatter(label_sorted_by_sbp[i], 0, original_sbp_sorted[i], zdir='y')#, marker='.', c='k')
                # ax.scatter(xs=0, ys=label2_sorted_by_sbp[i], zs=original_sbp_sorted[i], zdir='x',s=2)#, marker='.', c='k')
            # ax.scatter(t, abp_sig, ple_sig, c='b', marker='.', alpha=random.randint(25, 50) * 0.02, s=2)
            except:
                continue
        # plt.legend()
        plt.show()
        plt.close()

    if plot_shape_flag:
        plt.figure(figsize=(10, 10))
        print('plotting ppg shape graph... ')
        for i in tqdm(range(len(original_cycle))):
            # plt.plot(resampled_t, original_cycle[i], label=str(kmeans.labels_[i]), c=cm.jet(kmeans.labels_[i] / 5))
            plt.plot(resampled_t, original_cycle[i], label=str(kmeans.labels_[i]),
                     c=cm.jet((original_sbp[i] / np.max(original_sbp)) - 0.1), linewidth=0.8)
            # c=cm.jet(10), linewidth=0.8)
            # c=cm.jet(float(original_sbp[i]/np.max(original_sbp))))#, linewidth=1.0)
            plt.plot(resampled_t[check_points[i]], original_cycle[i][check_points[i]], 'kx', markersize=1)

        # plt.legend()
        plt.xlabel('Time (resampled)', fontsize=12)
        plt.ylabel('Plethysmo amplitude (normalized)', fontsize=12)
        plt.title('Plethysmo Single Cycle Classes', fontsize=14)
        plt.show()

    klist = [kmeans, kmeans2]
    if plot_class_map_flag:
        for idx, k in enumerate(klist):
            plt.figure(figsize=(8, 8))
            original_sbp_sorted = np.sort(np.squeeze(original_sbp))
            original_sbp_sorted_index = np.argsort(np.squeeze(original_sbp))

            labels = k.labels_[original_sbp_sorted_index]
            label_sorted_by_sbp = [labels[x] for x in original_sbp_sorted_index]
            print('plotting ppg class map graph... ')
            for i in tqdm(range(len(original_cycle))):
                if freq_info2[i] > 200:
                    continue
                plt.scatter(label_sorted_by_sbp[i], original_sbp_sorted[i], c=original_sbp_sorted[i],
                            marker='o', alpha=0.5, cmap=cm.jet, vmin=50, vmax=200, s=10)
            plt.title('Classification by Plethysmo Shape', fontsize=14)
            xticks = np.arange(0, cluster_num, 1)
            plt.xticks(xticks)
            plt.xlabel('Class', fontsize=12)
            plt.ylabel('Systolic Blood Pressure', fontsize=12)
            featrue_name = ['Shape', 'Frequency']
            plt.title('Clustered by PPG {} & SBP'.format(featrue_name[idx]), fontsize=14)
            plt.colorbar()
            plt.show()


ple_clustering(dataset,
               test_n=5000,
               cluster_num=10,
               check_point_n=1,
               plot_shape_flag=False,
               plot_class_map_flag=True,
               plot_3d_flag=True)


# kmeans = KMeans(n_clusters=5, random_state=0).fit(class_info_array)
# print(kmeans.labels_)

def plot_shape(original_cycle, original_map, check_points, resampled_t, kmeans):
    for i in range(len(original_cycle)):
        # plt.plot(resampled_t, original_cycle[i], label=str(kmeans.labels_[i]), c=cm.jet(kmeans.labels_[i] / 5))
        plt.plot(resampled_t, original_cycle[i], label=str(round(float(original_map[i]), 2)),
                 c=cm.tab10(kmeans.labels_[i] / 5))
        plt.plot(resampled_t[check_points[i]], original_cycle[i][check_points[i]], 'kx', markersize=1)

    plt.legend()
    plt.xlabel('Time (resampled)', fontsize=12)
    plt.ylabel('Plethysmo amplitude (normalized)', fontsize=12)
    plt.title('Plethysmo Single Cycle Classes', fontsize=14)
    plt.show()


def plot_class_map(original_cycle, original_map, kmeans):
    for i in range(len(original_cycle)):
        plt.scatter(int(kmeans.labels_[i]), original_map[i], c=original_map[i],
                    marker='.', alpha=0.5, cmap=cm.jet, vmin=50, vmax=180)
    plt.title('Classification by Plethysmo Shape', fontsize=14)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('MAP', fontsize=12)
    plt.title('Class & MAP', fontsize=14)
    plt.colorbar()
    plt.show()
