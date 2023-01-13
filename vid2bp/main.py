import os
import sys
import datetime
import wandb
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn

from dataset_loader import dataset_loader
from vid2bp.utils.train_utils import get_model, model_save, is_learning
from train import train
from validation import validation
from test import test
from pygame import mixer

mixer.init()
sound = mixer.Sound('bell-ringing-01c.wav')

torch.autograd.set_detect_anomaly(True)

with open('config/parameter.json') as f:
    json_data = json.load(f)
    param = json_data.get("parameters")
    channels = json_data.get("parameters").get("in_channels")
    out_channels = json_data.get("parameters").get("out_channels")
    hyper_param = json_data.get("hyper_parameters")
    wb = json_data.get("wandb")
    root_path = json_data.get("parameters").get("root_path")  # mimic uci containing folder
    data_path = json_data.get("parameters").get("dataset_path")  # raw mimic uci selection
    sampling_rate = json_data.get("parameters").get("sampling_rate")
    models = json_data.get("parameters").get("models")
    cases = json_data.get("parameters").get("cases")

print(sys.version)

# GPU Setting
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('----- GPU INFO -----\nDevice:', DEVICE)  # 출력결과: cuda
print('Count of using GPUs:', torch.cuda.device_count())
print('Current cuda device:', torch.cuda.current_device())
gpu_ids = list(map(str, list(range(torch.cuda.device_count()))))
total_gpu_memory = 0
for gpu_id in gpu_ids:
    total_gpu_memory += torch.cuda.get_device_properties("cuda:" + gpu_id).total_memory
print('Total GPU Memory :', total_gpu_memory, '\n--------------------')

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(125)
    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True
else:
    torch.manual_seed(125)
    print("cuda not available")


def main(model_name, dataset_name, in_channel, epochs, batch_size, scaler, wandb_on):
    # sound.play()

    # samp_rate = sampling_rate["60"]
    channel = channels[in_channel]
    # read_path = root_path + data_path[dataset_name][1]
    # TODO use hdf5 file for training Done

    """wandb setup"""
    # wandb_flag = True
    img_flag = False
    # wandb.init(project="VBPNet", entity="paperchae")

    """model setup"""
    model, loss, optimizer, scheduler = get_model(model_name, device=DEVICE)

    """dataset setup"""
    dataset = dataset_loader(dataset_name=dataset_name, channel=channel[0], batch_size=batch_size)
    train_cost_arr = []
    val_cost_arr = []
    test_cost_arr = []
    save_point = []

    print("start training")
    for epoch in range(epochs):
        if is_learning(val_cost_arr):
            '''train'''
            train_cost_arr.append(train(model, dataset[0], loss, optimizer, scheduler, epoch, scaler=not scaler))
            '''validation'''
            val_cost_arr.append(validation(model, dataset[1], loss, epoch, scaler=not scaler))
            if epoch % 1 == 0:
                ''' test model for each n epochs'''
                test_cost, plot_img = test(model, dataset[2], loss, epoch, scaler=not scaler, plot_target=False)
                test_cost_arr.append(test_cost)
                img_flag = True
            if epoch != 0:
                """ save model if train cost and val cost are lower than mean of previous epochs """
                if train_cost_arr[-1] < train_cost_arr[-2] and val_cost_arr[-1] < val_cost_arr[-2]:
                    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_point.append(current_time)
                    model_save(train_cost_arr, val_cost_arr, model, save_point, model_name, dataset_name)
            if wandb_on:
                ''' wandb logging '''
                if epoch == 0:
                    wandb.init(project="VBPNet", entity="paperchae")

                wandb.log({"Train_cost": train_cost_arr[-1],
                           "Val_cost": val_cost_arr[-1],
                           "Test_cost": test_cost_arr[-1]}, step=epoch)
                # if epoch!=0:
                if img_flag:
                    wandb.log({"Prediction": wandb.Image(plot_img)})
                    plot_img.close()
        else:
            print("model is not learning, stop training..")
            break
    """ plot Loss graph """
    t = np.array(range(len(train_cost_arr)))
    plt.title('Total epochs : {}'.format(len(t)))
    plt.plot(t, train_cost_arr, 'g-', label='Train Loss')
    plt.plot(t, val_cost_arr, 'b--', label='Validation Loss')
    plt.plot(t, test_cost_arr, 'r--', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    sound.play()

    print("training is done")


if __name__ == '__main__':
    main(model_name='BPNet', dataset_name='mimiciii', in_channel='second',
         epochs=200, batch_size=1024, scaler=False, wandb_on=True)
