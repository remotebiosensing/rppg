from NegPearsonLoss import Neg_Pearson
import numpy as np
from PhysNetED_BMVC import PhysNet_padding_Encoder_Decoder_MAX
import torch
import torchsummary
from tqdm import tqdm
import h5py
from torch.utils.data import DataLoader
import bvpdataset as bp
import time

start_save = time.time()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Available devices ', torch.cuda.device_count())
print('Current cuda device ', torch.cuda.current_device())
print(torch.cuda.get_device_name(device))

# device = torch.device('cuda:8') if torch.cuda.is_available() else torch.device('cpu')
# print(device)

model = PhysNet_padding_Encoder_Decoder_MAX().to(device)
torchsummary.summary(model, (3, 32, 128, 128))

model = torch.nn.DataParallel(model)

dataset = h5py.File("/media/hdd1/js_dataset/UBFC_PhysNet/UBFC_train_Data_delta_1_44.hdf5", 'r')
dataset = bp.dataset(data=dataset['output_video'], label=dataset['output_label'])
train_set, val_set = torch.utils.data.random_split(dataset,
                                                   [int(np.floor(len(dataset) * 0.7)),
                                                    int(np.ceil(len(dataset) * 0.3))])
train_loader = DataLoader(train_set, batch_size=32, shuffle=False)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_func = Neg_Pearson()
tmp_val_loss = 10.0

for epoch in range(20):
    running_loss = 0.0
    start = time.time()
    for i_batch, (data, label) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        data, label = data.to(device), label.to(device)
        output = model(data)
        label = (label - torch.mean(label)) / torch.std(label)
        rPPG = (output[0] - torch.mean(output[0])) / torch.std(output[0])
        loss = loss_func(rPPG, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('Train : [%d, %5d] loss: %.3f' % (epoch + 1, i_batch + 1, running_loss / 32))
    print("time :", time.time() - start)
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for v_batch, (v_data, v_label) in enumerate(val_loader):
            v_data, v_label = v_data.to(device), v_label.to(device)
            v_output = model(v_data)
            v_label = (v_label - torch.mean(v_label)) / torch.std(v_label)
            v_rPPG = (v_output[0] - torch.mean(v_output[0])) / torch.std(v_output[0])
            v_loss = loss_func(v_rPPG, v_label)
            val_loss += v_loss.item()
        print('Val : [%d, %5d] loss: %.3f' % (epoch + 1, v_batch + 1, val_loss / 32))
        if tmp_val_loss > (val_loss / 32):
            checkpoint = {'Epoch': epoch,
                          'state_dict': model.state_dict(),
                          'optimizer': optimizer.state_dict()}
            torch.save(checkpoint,
                       '/home/js/Desktop/PhysNet/model/PhysNet_UBFC ' + time.ctime(start_save) + '.pth')
            tmp_val_loss = val_loss / 32
            print("Update tmp : " + str(tmp_val_loss))
    model.train()
print("Finish Train")


