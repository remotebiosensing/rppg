import model
import bvpdataset
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

transform = transforms.Compose([transforms.ToTensor()])
dataset = bvpdataset.bvpdataset(
    data_path="/mnt/a7930c08-d429-42fa-a09e-15291e166a27/BVP_js/DATASET_2/M/subject_total/subject_speed_up.npz",
    transform=transform)
train_set, val_set = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8),int(len(dataset)*0.2+1)],generator=torch.Generator().manual_seed(1))
train_loader = DataLoader(train_set, batch_size=128, shuffle=False)
val_loader = DataLoader(val_set, batch_size=128, shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Availabel devices', torch.cuda.device_count())
print('Current cuda device', torch.cuda.current_device())
print(torch.cuda.get_device_name(device))

GPU_NUM = 0
torch.cuda.set_device(GPU_NUM)

model = model.DeepPhys(in_channels=3, out_channels=32, kernel_size=3).to(device)
writer.add_graph(model)
writer.close()
MSEloss = torch.nn.MSELoss()
Adadelta = optim.Adadelta(model.parameters(), lr=1)
for epoch in range(1000000):
    #running_loss = 0.0
    for i_batch, (avg, mot, lab) in enumerate(train_loader):

        Adadelta.zero_grad()



        avg, mot, lab = avg.to(device), mot.to(device), lab.to(device)
        output = model(avg, mot)
        loss = MSEloss(output, lab)
        if torch.isnan(loss):
            continue
        loss.backward()
        #model.float()
        #torch.nn.utils.clip_grad_norm(model.parameters(),4)
        Adadelta.step()
        if i_batch % 10 == 0:
            print("epoch {} batch {} loss {}".format(epoch, i_batch, loss.data))

    if epoch % 5 == 0:
        with torch.no_grad():
            val_loss = 0.0
            for k, (avg, mot, lab) in enumerate(val_loader):
                avg, mot, lab = avg.to(device), mot.to(device), lab.to(device)
                val_output = model(avg, mot)
                v_loss = MSEloss(val_output, lab)
                if torch.isnan(v_loss):
                    continue
                val_loss += v_loss
        print("validation loss {}".format(val_loss))


            #torch.save([model1, model2], os.path.join(save_model_path, 'pretrained_model.pt'))

        #running_loss += loss.item()
   # print('[%d] loss: %.3f [%d]' %  (epoch + 1, running_loss, i_batch))



print('Finished Training')
