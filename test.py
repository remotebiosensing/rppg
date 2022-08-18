import os
import torch
from torch.utils.data import DataLoader
from preprocessing import MIMICdataset, customdataset
from matplotlib import pyplot as plt
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model_test():
    root_path = '/home/paperc/PycharmProjects/BPNET/dataset/mimic-database-1.0.0/'

    t_path2 = '039/03900004'
    test_record = MIMICdataset.read_record(root_path + t_path2)
    chunk_size = 10
    test_abp, test_ple = MIMICdataset.sig_slice(signals=test_record.p_signal, size=chunk_size)

    test_dataset = customdataset.CustomDataset(x_data=test_ple[1], y_data=test_abp[1])
    test_loader = DataLoader(test_dataset, batch_size=7500, shuffle=False)

    dataiter = iter(test_loader)
    seq, labels = dataiter.__next__()

    print(seq)
    print(len(seq))
    #

    Model_PATH = '/home/paperc/PycharmProjects/BPNET/weights/'

    load_model = torch.load(Model_PATH + 'model.pt')  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
    load_model.load_state_dict(torch.load(Model_PATH + 'model_state_dict.pt'))  # state_dict를 불러 온 후, 모델에 저장

    with torch.no_grad():
        # X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
        X_test = seq
        Y_test = labels

        print(type(Y_test))
        # Y_test = mnist_test.test_labels.to(device)

        prediction = load_model(X_test)
        print(type(prediction))
        prediction = np.reshape(prediction.cpu().numpy().tolist(), (7500, 1))[:375]
        Y_test = Y_test.cpu().numpy().tolist()[:375]
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

        plt.title('prediction')
        plt.show()


model_test()
