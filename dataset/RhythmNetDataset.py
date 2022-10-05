import torch
import numpy as np
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset
from utils.signal_utils import read_target_data, calculate_hr, get_hr_data

ImageFile.LOAD_TRUNCATED_IMAGES = True


class RhythmNetDataset(Dataset):
    """
        Dataset class for RhythmNet
    """
    # The data is now the SpatioTemporal Maps instead of videos

    def __init__(self, st_maps_path, target_signal_path):
        self.H = 180
        self.W = 180
        self.C = 3
        # self.video_path = data_path
        self.st_maps_path = st_maps_path
        # self.resize = resize
        self.target_path = target_signal_path
        self.maps = None

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        # Maybe add more augmentations
        # self.augmentation_pipeline = albumentations.Compose(
        #     [
        #         albumentations.Normalize(
        #             mean, std, max_pixel_value=255.0, always_apply=True
        #         )
        #     ]
        # )

    def __len__(self):
        return len(self.st_maps_path)

    def __getitem__(self, index):
        # identify the name of the video file so as to get the ground truth signal
        self.video_file_name = self.st_maps_path[index].split('/')[-1].split('.')[0]
        # targets, timestamps = read_target_data(self.target_path, self.video_file_name)
        # sampling rate is video fps (check)

        # Load the maps for video at 'index'
        self.maps = np.load(self.st_maps_path[index])
        map_shape = self.maps.shape
        self.maps = self.maps.reshape((-1, map_shape[3], map_shape[1], map_shape[2]))

        # target_hr = calculate_hr(targets, timestamps=timestamps)
        # target_hr = calculate_hr_clip_wise(map_shape[0], targets, timestamps=timestamps)
        target_hr = get_hr_data(self.video_file_name)
        # To check the fact that we dont have number of targets greater than the number of maps
        # target_hr = target_hr[:map_shape[0]]
        self.maps = self.maps[:target_hr.shape[0], :, :, :]
        return {
            "st_maps": torch.tensor(self.maps, dtype=torch.float),
            "target": torch.tensor(target_hr, dtype=torch.float)
        }