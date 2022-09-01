import os
import torch
from torch.utils.data import DataLoader
from preprocessing import MIMICdataset, customdataset
from matplotlib import pyplot as plt
import numpy as np
from preprocessing.utils import math_module
from scipy import signal
import json

with open('/home/paperc/PycharmProjects/BPNET/parameter.json') as f:
    json_data = json.load(f)
    param = json_data.get("parameters")
    orders = json_data.get("parameters").get("in_channels")
    test_orders = json_data.get("parameters").get("test_channels")
    sampling_rate = json_data.get("parameters").get("sampling_rate")


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

order = orders["first"]
# degree = order - 1
test_order = test_orders["third"]
degree = test_order - 1
# test_len = int(param["down_sample"] / 60) * 2
test_len = int(param["chunk_size"] / 125) * 2

def model_test():
    root_path = '/home/paperc/PycharmProjects/BPNET/dataset/mimic-database-1.0.0/'

    # t_path2 = '472/47200003'
    # t_path2 = '039/03900011'
    t_path2 = '055/05500100'
    test_record = MIMICdataset.read_record(root_path + t_path2)
    print(np.shape(test_record))
    chunk_num = 10

    if degree == 0:
        test_ple_total, test_abp = MIMICdataset.signal_slicing(signals=test_record, chunk_num=chunk_num)
    elif degree == 1:
        test_ple, test_abp = MIMICdataset.signal_slicing(signals=test_record, chunk_num=chunk_num)
        test_ple_first = math_module.diff_np(test_ple)
        test_ple_total = test_ple_first
    elif degree == 2:
        test_ple, test_abp = MIMICdataset.signal_slicing(signals=test_record, chunk_num=chunk_num)
        test_ple_first = math_module.diff_np(test_ple)
        test_ple_second = math_module.diff_np(test_ple_first)
        test_ple_total = test_ple_second
    elif degree == 3:
        test_ple, test_abp = MIMICdataset.signal_slicing(signals=test_record, chunk_num=chunk_num)
        test_ple_first = math_module.diff_np(test_ple)

        test_ple_total = math_module.diff_channels_aggregator(test_ple, test_ple_first)
        test_abp = signal.resample(test_abp, int(param["chunk_size"] / 125) * sampling_rate["60"])
    elif degree == 4:
        test_ple, test_abp = MIMICdataset.signal_slicing(signals=test_record, chunk_num=chunk_num)
        test_ple_first = math_module.diff_np(test_ple)
        test_ple_second = math_module.diff_np(test_ple_first)

        test_ple_total = math_module.diff_channels_aggregator(test_ple, test_ple_second)
    elif degree == 5:
        test_ple, test_abp = MIMICdataset.signal_slicing(signals=test_record, chunk_num=chunk_num)
        test_ple_first = math_module.diff_np(test_ple)
        test_ple_second = math_module.diff_np(test_ple_first)

        test_ple_total = math_module.diff_channels_aggregator(test_ple_first, test_ple_second)
    else:
        test_ple, test_abp = MIMICdataset.signal_slicing(signals=test_record, chunk_num=chunk_num)
        test_ple_first = math_module.diff_np(test_ple)
        test_ple_second = math_module.diff_np(test_ple_first)

        test_ple_total = math_module.diff_channels_aggregator(test_ple, test_ple_first, test_ple_second)

    print('test_ple shape :', np.shape(test_ple_total))
    print('test_abp shape :', np.shape(test_abp))

    test_dataset = customdataset.CustomDataset(x_data=test_ple_total[0:2], y_data=test_abp[0:2])
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    dataiter = iter(test_loader)
    seq, labels = dataiter.__next__()

    Model_PATH = param["save_path"]

    # load_model = torch.load(Model_PATH + 'model_' + str(order - 1) + '_NegMAE_baseline.pt')  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
    load_model = torch.load(Model_PATH + 'model_110_NegMAE_newmodel_temp.pt')  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
    # load_model.load_state_dict(torch.load(Model_PATH + 'model_state_dict.pt'))  # state_dict를 불러 온 후, 모델에 저장

    with torch.no_grad():
        X_test = seq
        Y_test = labels

        prediction = load_model(X_test)

        print('np.shape(X_test) :', np.shape(X_test))
        print('np.shape(prediction) :', np.shape(prediction))

        prediction = torch.squeeze(prediction).cpu().numpy()[0][30:test_len + 30]
        Y_test = Y_test.cpu().numpy()[0][30:test_len + 30]

        time = np.array(range(len(prediction)))
        fig, ax1 = plt.subplots()

        ax1.plot(time, prediction, color="red")
        ax1.tick_params(axis='y', labelcolor="red")

        ax2 = ax1.twinx()
        ax2.plot(time, Y_test, color="blue")
        ax2.tick_params(axis='y', labelcolor="blue")

        plt.title('F + F\' MAE * Neg_Pearson_Loss prediction')
        plt.show()


model_test()
