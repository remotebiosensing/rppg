import torch
from matplotlib import pyplot as plt

class test_model:

    def __init__(self, name, models, test_loader, criterion, model_path, num_epochs, device):
        self.name = name
        self.model = models
        self.num_epochs = num_epochs
        self.model_path = model_path
        self.criterion = criterion
        self.test_loader = test_loader
        self.device = device

        self.model.to(device)

        with torch.no_grad():
            val_output = []
            for k, (avg, mot, lab) in enumerate(test_loader):
                if self.name.find("TS") is not -1:
                    if avg.shape[0] %2 is 1:
                        continue
                avg, mot, lab = avg.to(device), mot.to(device), lab.to(device)
                val_output.append(self.model(avg, mot).cpu().clone().numpy()[0][0])

        target = test_loader.dataset.label.tolist()
        plt.rcParams["figure.figsize"] = (14, 5)
        plt.plot(range(len(val_output[:300])), val_output[:300], label='inference')
        plt.plot(range(len(test_loader.dataset.label[:300])), target[:300], label='target')
        plt.legend(fontsize='x-large')
        plt.show()