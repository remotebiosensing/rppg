import model
import bvpdataset
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Availabel devices', torch.cuda.device_count())
print('Current cuda device', torch.cuda.current_device())
print(torch.cuda.get_device_name(device))

GPU_NUM = 0
torch.cuda.set_device(GPU_NUM)

transform = transforms.Compose([transforms.ToTensor()])
dataset = bvpdataset.bvpdataset(
    data_path="subject_test.npz",
    transform=transform)
test_loader = DataLoader(dataset, batch_size=1, shuffle=False)


model = model.DeepPhys(in_channels=3, out_channels=32, kernel_size=3).cuda()
checkpoint = torch.load("checkpoint.pth")
model.load_state_dict(checkpoint['state_dict'])

with torch.no_grad():
    val_output = []
    for k, (avg, mot, lab) in enumerate(test_loader):
        avg, mot, lab = avg.to(device), mot.to(device), lab.to(device)
        val_output.append(model(avg, mot).cpu().clone().numpy()[0][0])

print(val_output)