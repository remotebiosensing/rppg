import model
import bvpdataset
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class test_model:
    def __init__(self, models, test_loader, criterion, optimizers, model_path, num_epochs, device):
        self.model = models
        self.num_epochs = num_epochs
        self.model_path = model_path
        self.optimizers = optimizers
        self.criterion = criterion
        self.test_loader = test_loader
        self.device = device

        self.model.to(device)

        with torch.no_grad():
            val_output = []
            for k, (avg, mot, lab) in enumerate(test_loader):
                if self.model.find("TS") is not -1:
                    if avg.shape[0] %2 is 1:
                        continue
                avg, mot, lab = avg.to(device), mot.to(device), lab.to(device)
                val_output.append(self.model(avg, mot).cpu().clone().numpy()[0][0])

        print(val_output)