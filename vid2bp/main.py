import os
import sys
import datetime
import wandb
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.cuda as cuda
import torch.backends.cudnn as cudnn

from dataset_loader import dataset_loader
from utils.train_utils import get_model, model_save, is_learning
from train import train
from validation import validation
from test import test
from pygame import mixer
from vid2bp import stage2_main
import pandas as pd

mixer.init()
sound = mixer.Sound('bell-ringing-01c.wav')

''' warning: do not turn on set_detect_anomaly(True) when training, only for debugging '''
# torch.autograd.set_detect_anomaly(True)

with open('config/parameter.json') as f:
    json_data = json.load(f)
    # param = json_data.get("parameters")
    channels = json_data.get("parameters").get("in_channels")
    gender = json_data.get("parameters").get("gender")
    # out_channels = json_data.get("parameters").get("out_channels")
    # hyper_param = json_data.get("hyper_parameters")
    # wb = json_data.get("wandb")
    root_path = json_data.get("parameters").get("root_path")  # mimic uci containing folder
    data_path = json_data.get("parameters").get("dataset_path")  # raw mimic uci selection
    # sampling_rate = json_data.get("parameters").get("sampling_rate")
    models = json_data.get("parameters").get("models")
    # cases = json_data.get("parameters").get("cases")

print(sys.version)

# GPU Setting
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('----- GPU INFO -----\nDevice:', DEVICE)
print('Count of using GPUs:', torch.cuda.device_count())
print('Current cuda device:', torch.cuda.current_device())
gpu_ids = list(map(str, list(range(torch.cuda.device_count()))))
total_gpu_memory = 0
for gpu_id in gpu_ids:
    total_gpu_memory += torch.cuda.get_device_properties("cuda:" + gpu_id).total_memory
print('Total GPU Memory :', total_gpu_memory, '\n--------------------')

if torch.cuda.is_available():
    random_seed = 125
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    cuda.manual_seed(random_seed)
    cuda.allow_tf32 = True
    # torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    cudnn.enabled = True
    cudnn.deterministic = True  # turn on for reproducibility ( if turned on, slow down training )
    cudnn.benchmark = False
    cudnn.allow_tf32 = True
else:
    print("cuda not available")


def main(model_name, dataset_name, in_channel, epochs, batch_size, wandb_on, gen, normalized):
    # sound.play()
    patient_info = pd.read_csv(
        '/home/paperc/PycharmProjects/dataset/BPNet_' + dataset_name + '/additional2023131_normalized/patient_data.csv')
    channel_info = channels[in_channel]
    gender_info = gender[gen]

    # TODO use hdf5 file for training Done

    """wandb setup"""
    img_flag = False

    """model setup"""
    model, loss, optimizer, scheduler = get_model(model_name, channel=channel_info[0], device=DEVICE, stage=1)
    # model_2, loss_2, optimizer_2, scheduler_2 = get_model(model_name, device=DEVICE, stage=2)

    """dataset setup"""
    dataset = dataset_loader(dataset_name=dataset_name, in_channel=channel_info[-1], batch_size=batch_size,
                             device=DEVICE,
                             gender=gender_info[-1], normalized=normalized)
    train_cost_arr = []
    val_cost_arr = []
    test_cost_arr = []
    save_point = []

    print('----- model info -----')
    print(model)
    print("start training")
    for epoch in range(epochs):
        # if is_learning(val_cost_arr):
        '''train'''
        train_cost_arr.append(train(model, dataset[0], loss, optimizer, scheduler, epoch))
        '''validation'''
        val_cost_arr.append(validation(model, dataset[1], loss, epoch))
        if epoch % 1 == 0:
            ''' test model for each n epochs'''
            test_cost, plot_img = test(model, dataset[2], loss, epoch, plot_scaled=False,
                                       patient_information=patient_info)
            test_cost_arr.append(test_cost)
            img_flag = True
        if epoch != 0:
            """ save model if train cost and val cost are lower than mean of previous epochs """
            # if train_cost_arr[-1] < train_cost_arr[-2] and val_cost_arr[-1] < val_cost_arr[-2]:
            # print(np.min(train_cost_arr[:-1]), np.min(val_cost_arr[:-1]))
            # print(train_cost_arr[-1], val_cost_arr[-1])
            if train_cost_arr[-1] < np.min(train_cost_arr[:-1]) and val_cost_arr[-1] < np.min(val_cost_arr[:-1]):
                current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_point.append(current_time)
                best_model_path = model_save(train_cost_arr, val_cost_arr, model, save_point, model_name,
                                             dataset_name, channel_info[-2], gender_info[-1])
        if wandb_on:
            ''' wandb logging '''
            if epoch == 0:
                wandb.init(project="VBPNet", entity="paperchae",
                           name=str(gen) + '_' + channel_info[1] + '_' + str(batch_size))

            wandb.log({"Train_cost": train_cost_arr[-1],
                       "Val_cost": val_cost_arr[-1],
                       "Test_cost": test_cost_arr[-1]}, step=epoch)
            # if epoch!=0:
            if img_flag:
                wandb.log({"Prediction": wandb.Image(plot_img)})
                plot_img.close()
        # else:
        #     print("model is not learning, stop training..")
        #     print("best model path : {}".format(best_model_path))
        #     break
    """ plot stage 1 Loss graph """
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
    print("training stage 1 is done")
    print("best model path : {}".format(best_model_path))
    wandb.finish()

    '''stage 2'''
    # stage2_main.main2(best_model_path, model, model_2,
    #                   loss_2, optimizer_2, scheduler_2, epochs, dataset, wandb_on, model_name, dataset_name)


if __name__ == '__main__':
    batch = [256, 512, 1024, 2048, 4096]
    # gender = [0, 1, 2]
    gender_list = ["Total", "Male", "Female"]
    # ch = ['P', 'V', 'A', 'PV', 'PA', 'VA', 'PVA']
    ch = ['P', 'PV', 'PA', 'PVA']

    # 0 zeroth, 0 first, 0 second, 1 zeroth, 1 first, 1 second, 2 zeroth, 2 first, 2 second
    # for g in gender:
    #     for c in ch:
    #         for b in batch:
    #             main(model_name='BPNet', dataset_name='mimiciii', in_channel=c,
    #                  epochs=30, batch_size=b, wandb_on=True, gender=g, normalized=True)
    for c in ch:
        main(model_name='BPNet', dataset_name='mimiciii', in_channel=c, epochs=30, batch_size=512,
             wandb_on=True, gen="Total", normalized=True)
    # main(model_name='BPNet', dataset_name='mimiciii', in_channel='VA', epochs=30, batch_size=2048, wandb_on=True,
    #      gen="Total", normalized=True)
