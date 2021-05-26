import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class bvpdataset(Dataset):
    def __init__(self, A, M, T):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.a = A
        self.m = M
        self.label = T.reshape(-1, 1)

    def __getitem__(self, index, dtype=torch.float):
        if torch.is_tensor(index):
            index = index.tolist()

        # norm_img = torch.tensor(np.transpose(self.a[index]), dtype=dtype)
        # mot_img = torch.tensor(np.transpose(self.m[index]), dtype=dtype)
        norm_img = torch.tensor(self.a[index], dtype=dtype)
        mot_img = torch.tensor(self.m[index], dtype=dtype)
        label = torch.tensor(self.label[index], dtype=dtype)

        return norm_img, mot_img, label

    def __len__(self):
        return len(self.label)
