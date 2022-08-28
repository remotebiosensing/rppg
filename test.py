import os
import torch
from torch.utils.data import DataLoader
from preprocessing import MIMICdataset, customdataset
from matplotlib import pyplot as plt
import numpy as np
from preprocessing.utils import math_module
from scipy import signal

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_test():
    root_path = '/home/paperc/PycharmProjects/BPNET/dataset/mimic-database-1.0.0/'

    t_path2 = '055/05500011'
    test_record = MIMICdataset.read_record(root_path + t_path2)
    chunk_size = 10
    test_ple, test_abp = MIMICdataset.signal_slicing(signals=test_record, chunk_num=chunk_size)

    ple = []
    abp = []

    for i in range(len(test_ple)):
        ple.append(signal.resample(test_ple[i], 1800))
        abp.append(signal.resample(test_abp[i], 1800))

    test_ple = np.asarray(ple)
    test_abp = np.asarray(abp)

    test_ple_first = math_module.diff_np(test_ple)
    test_ple_second = math_module.diff_np(test_ple_first)

    test_total = math_module.diff_channels_aggregator(test_ple, test_ple_first, test_ple_second)
    print('test_total shape :', np.shape(test_total))

    test_dataset = customdataset.CustomDataset(x_data=test_total[0:2], y_data=test_abp[0:2])
    test_loader = DataLoader(test_dataset, batch_size=7500, shuffle=False)

    dataiter = iter(test_loader)
    seq, labels = dataiter.__next__()

    Model_PATH = '/home/paperc/PycharmProjects/BPNET/weights/'

    load_model = torch.load(Model_PATH + 'model_derivative_test.pt')  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
    # load_model.load_state_dict(torch.load(Model_PATH + 'model_state_dict.pt'))  # state_dict를 불러 온 후, 모델에 저장

    with torch.no_grad():
        # X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
        X_test = seq
        Y_test = labels

        prediction = load_model(X_test)

        print('np.shape(X_test) :', np.shape(X_test))
        print('np.shape(prediction) :', np.shape(prediction))

        prediction = torch.squeeze(prediction).cpu().numpy()[0][:180]
        Y_test = Y_test.cpu().numpy()[0][:180]
        # prediction = np.reshape(prediction.cpu().numpy().tolist(), (15000, 1))[:500]
        # Y_test = Y_test.cpu().numpy().tolist()[:500]
        # print(prediction[:10])
        # correct_prediction = torch.argmax(prediction, 1) == Y_test
        # accuracy = correct_prediction.float().mean()
        # print('Accuracy :', accuracy.item())

        time = np.array(range(len(prediction)))
        fig, ax1 = plt.subplots()

        ax1.plot(time, prediction, color="red")
        ax1.tick_params(axis='y', labelcolor="red")

        ax2 = ax1.twinx()
        ax2.plot(time, Y_test, color="blue")
        ax2.tick_params(axis='y', labelcolor="blue")

        plt.title('RMSE * Neg_Pearson_Loss prediction')
        plt.show()


model_test()
