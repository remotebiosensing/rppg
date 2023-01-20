import h5py
import numpy as np
import signal_utils as su
import dataset.dataset_loader as dl
import matplotlib.pyplot as plt


def dataset_loader(data_root_path: str = '/media/hdd1/dy/dataset/TEST_UBFC_2023-01-19_0.hdf5'):
    video = []
    keypoint = []
    label = []
    flag = True
    with h5py.File(data_root_path, 'r') as f:
        dl.h5_tree(f)
        for key in f.keys():
            if flag:
                flag = False
                continue
            video.append(f[key]['raw_video'][:])
            keypoint.append(f[key]['keypoint'][:])
            label.append(f[key]['preprocessed_label'][:])
            break
        return np.asarray(video), np.asarray(keypoint, dtype=np.float32), np.asarray(label)

def main(data_root_path):
    # get video, keypoint, label
    video, keypoint, label = dataset_loader(data_root_path='/media/hdd1/dy/dataset/TEST_UBFC_2023-01-19_0.hdf5')

    for target_video, target_keypoint, target_label in zip(video, keypoint, label):

        cycle_length = su.get_cycle_length(target_label, fs=29.26)

        # replace -n by 0
        target_keypoint = su.replace_minus_by_zero(target_keypoint)
        # get bounding box of face
        target_keypoint = np.mean(target_keypoint, axis=0, dtype=int)

        x_p = target_keypoint[::2]
        y_p = target_keypoint[1::2]

        forehead_video = target_video[:, y_p[0]:y_p[1], x_p[0]:x_p[1], :]
        left_cheek_video = target_video[:, y_p[2]:y_p[3], x_p[2]:x_p[3], :]
        right_cheek_video = target_video[:, y_p[4]:y_p[5], x_p[4]:x_p[5], :]

        forehead_mean_rgb = su.get_rgb_mean(forehead_video)
        left_cheek_mean_rgb = su.get_rgb_mean(left_cheek_video)
        right_cheek_mean_rgb = su.get_rgb_mean(right_cheek_video)
        target_video_mean_rgb = su.get_rgb_mean(target_video)

        target_video_pos = su.pos(target_video_mean_rgb, 29.26, 29)
        forehead_pos = su.pos(forehead_mean_rgb, 29.26, 29)
        left_cheek_pos = su.pos(left_cheek_mean_rgb, 29.26, 29)
        right_cheek_pos = su.pos(right_cheek_mean_rgb, 29.26, 29)
        print('Total video')
        print('video - label', su.get_correlation(target_video_pos, target_label))
        print('forehead - label', su.get_correlation(forehead_pos, target_label))
        print('left_cheek - label', su.get_correlation(left_cheek_pos, target_label))
        print('right_cheek - label', su.get_correlation(right_cheek_pos, target_label))
        print('video - forehead', su.get_correlation(target_video_pos, forehead_pos))
        print('video - left_cheek', su.get_correlation(target_video_pos, left_cheek_pos))
        print('video - right_cheek', su.get_correlation(target_video_pos, right_cheek_pos))
        print('forehead - left_cheek : ', su.get_correlation(forehead_pos, left_cheek_pos))
        print('forehead - right_cheek : ', su.get_correlation(forehead_pos, right_cheek_pos))
        print('left_cheek - right_cheek : ', su.get_correlation(left_cheek_pos, right_cheek_pos))
        print('=' * 100)
        print('Sliced video')
        sliced_video_pos = su.signal_slicer(target_video_pos, cycle_length, 1, 0)
        sliced_forehead_pos = su.signal_slicer(forehead_pos, cycle_length, 1, 0)
        sliced_left_cheek_pos = su.signal_slicer(left_cheek_pos, cycle_length, 1, 0)
        sliced_right_cheek_pos = su.signal_slicer(right_cheek_pos, cycle_length, 1, 0)
        sliced_label = su.signal_slicer(target_label, cycle_length, 1, 0)
        # forehead_video = sc.signal_slicer(forehead_video, cycle_length, 1, 0)
        # left_cheek_video = sc.signal_slicer(left_cheek_video, cycle_length, 1, 0)
        # right_cheek_video = sc.signal_slicer(right_cheek_video, cycle_length, 1, 0)

        video_label = []
        forehead_label = []
        left_cheek_label = []
        right_cheek_label = []
        video_forehead = []
        video_left_cheek = []
        video_right_cheek = []
        forehead_left_cheek = []
        forehead_right_cheek = []
        left_cheek_right_cheek = []

        for target_video_pos, target_forehead_pos, target_left_cheek_pos, target_right_cheek_pos, target_label_ppg in zip(
                sliced_video_pos[:-1],
                sliced_forehead_pos[:-1],
                sliced_left_cheek_pos[:-1],
                sliced_right_cheek_pos[:-1],
                sliced_label[:-1]):
            video_label.append(su.get_correlation(target_video_pos, target_label_ppg))
            forehead_label.append(su.get_correlation(target_forehead_pos, target_label_ppg))
            left_cheek_label.append(su.get_correlation(target_left_cheek_pos, target_label_ppg))
            right_cheek_label.append(su.get_correlation(target_right_cheek_pos, target_label_ppg))
            video_forehead.append(su.get_correlation(target_video_pos, target_forehead_pos))
            video_left_cheek.append(su.get_correlation(target_video_pos, target_left_cheek_pos))
            video_right_cheek.append(su.get_correlation(target_video_pos, target_right_cheek_pos))
            forehead_left_cheek.append(su.get_correlation(target_forehead_pos, target_left_cheek_pos))
            forehead_right_cheek.append(su.get_correlation(target_forehead_pos, target_right_cheek_pos))
            left_cheek_right_cheek.append(su.get_correlation(target_left_cheek_pos, target_right_cheek_pos))

        print('video - label(one cycle)', np.mean(np.asarray(video_label)))
        print('forehead - label(one cycle)', np.mean(np.asarray(forehead_label)))
        print('left_cheek - label(one cycle)', np.mean(np.asarray(left_cheek_label)))
        print('right_cheek - label(one cycle)', np.mean(np.asarray(right_cheek_label)))
        print('video - forehead(one cycle)', np.mean(np.asarray(video_forehead)))
        print('video - left_cheek(one cycle)', np.mean(np.asarray(video_left_cheek)))
        print('video - right_cheek(one cycle)', np.mean(np.asarray(video_right_cheek)))
        print('forehead - left_cheek(one cycle) : ', np.mean(np.asarray(forehead_left_cheek)))
        print('forehead - right_cheek(one cycle) : ', np.mean(np.asarray(forehead_right_cheek)))
        print('left_cheek - right_cheek(one cycle) : ', np.mean(np.asarray(left_cheek_right_cheek)))
        print('=' * 100)

        # forehead_mean_green = su.get_channels_mean(target_forehead_video, 1)
        # left_cheek_mean_green = su.get_channels_mean(target_left_cheek_video, 1)
        # right_cheek_mean_green = su.get_channels_mean(target_right_cheek_video, 1)
        #
        # print('----------------------------------------')
        # print('forehead - left_cheek (G): ', su.get_correlation(forehead_mean_green, left_cheek_mean_green))
        # print('forehead - right_cheek (G): ', su.get_correlation(forehead_mean_green, right_cheek_mean_green))
        # print('left_cheek - right_cheek (G): ', su.get_correlation(left_cheek_mean_green, right_cheek_mean_green))
        # plt.imshow(target_forehead_video[0])
        # plt.title('forehead')
        # plt.show()
        # plt.imshow(target_left_cheek_video[0])
        # plt.title('left_cheek')
        # plt.show()
        # plt.imshow(target_right_cheek_video[0])
        # plt.title('right_cheek')
        # print('forehead - left_cheek (RGB): ', su.get_correlation(forehead_mean_rgb, left_cheek_mean_rgb))
        # print('forehead - right_cheek (RGB): ', su.get_correlation(forehead_mean_rgb, right_cheek_mean_rgb))
        # print('left_cheek - right_cheek (RGB): ', su.get_correlation(left_cheek_mean_rgb, right_cheek_mean_rgb))
        # print('----------------------------------------')
        # # get timewise mean of channels
        # forehead_mean_rgb = su.get_rgb_mean(forehead_video)
        # forehead_mean_green = su.get_channels_mean(forehead_video, 1)
        #
        # L_cheek_mean_rgb = su.get_rgb_mean(left_cheek_video)
        # L_cheek_mean_green = su.get_channels_mean(left_cheek_video, 1)
        #
        # R_cheek_mean_rgb = su.get_rgb_mean(right_cheek_video)
        # R_cheek_mean_green = su.get_channels_mean(right_cheek_video, 1)
        #
        # # get correlation of forehead, left_cheek, right_cheek
        # forehead_L_cheek_correlation = su.get_correlation(forehead_mean_rgb, L_cheek_mean_rgb)
        # forehead_R_cheek_correlation = su.get_correlation(forehead_mean_rgb, R_cheek_mean_rgb)
        # L_cheek_R_cheek_correlation = su.get_correlation(L_cheek_mean_rgb, R_cheek_mean_rgb)


if __name__ == '__main__':
    main(data_root_path='/media/hdd1/dy/dataset/TEST_UBFC_2023-01-19_0.hdf5')
