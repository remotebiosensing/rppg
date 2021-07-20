import h5py
import scipy.signal
import torch
import bvpdataset as bp
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import butter
from torch.utils.data import DataLoader
from PhysNetED_BMVC import PhysNet_padding_Encoder_Decoder_MAX

device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
print('Available devices ', torch.cuda.device_count())
print('Current cuda device ', torch.cuda.current_device())
print(torch.cuda.get_device_name(device))

model = PhysNet_padding_Encoder_Decoder_MAX().to(device)
model = torch.nn.DataParallel(model, device_ids=[4, 5, 6, 7])

checkpoint = torch.load("/home/js/Desktop/PhysNet/model" + "/PhysNet_UBFC Mon Jul 19 15:32:43 2021.pth")
model.load_state_dict(checkpoint['state_dict'])
model.eval()
for n in range(45, 50):
    dataset = h5py.File("/media/hdd1/js_dataset/UBFC_PhysNet/UBFC_test_Data_" + str(n) + ".hdf5", 'r')
    test_set = bp.dataset(data=dataset['output_video'], label=dataset['output_label'])
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    with torch.no_grad():
        total_label = []
        total_rPPG = []
        for t_batch, (data, label) in enumerate(test_loader):
            data, label = data.to(device), label.to(device)
            output = model(data)
            label = (label - torch.mean(label)) / torch.std(label)
            rPPG = (output[0] - torch.mean(output[0])) / torch.std(output[0])
            total_label.extend(label[0].tolist())
            total_rPPG.extend(rPPG[0].tolist())
        print("End : Inference")
    fs = 30
    low = 0.75 / (0.5 * fs)
    high = 2.5 / (0.5 * fs)
    [b_pulse, a_pulse] = butter(1, [low, high], btype='bandpass')
    total_rPPG = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(total_rPPG))
    total_label = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(total_label))
    corr = np.corrcoef(total_rPPG, total_label)
    print(corr)
    # matplotlib---------------------------------------------------------------------
    k = 0
    for i in range(0, 5):
        plt.rcParams["figure.figsize"] = (14, 5)
        plt.plot(range(len(total_label[k:k + 300])), total_label[k:k + 300], label='target')
        plt.plot(range(len(total_rPPG[k:k + 300])), total_rPPG[k:k + 300], label='inference')
        plt.legend(fontsize='x-large')
        plt.show()
        k += 300
