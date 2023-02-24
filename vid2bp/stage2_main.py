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
from vid2bp.utils.train_utils import get_model, model_save, is_learning
from train import train
from validation import validation
from test import test
from pygame import mixer

epochs = 200
stage2_train_cost_arr = []
stage2_val_cost_arr = []
stage2_test_cost_arr = []
stage2_save_point = []
# dataset = dataset_loader(dataset_name=dataset_name, channel=channel[0], batch_size=batch_size, device=DEVICE)


def main2(stage1_best_model_path, stage1_model, stage2_model,
                stage2_loss, stage2_optimizer, stage2_scheduler, epochs, dataset, wandb_on, model_name, dataset_name):
    stage1_model.load_state_dict(torch.load(stage1_best_model_path))

    for epoch in range(epochs):
        if is_learning(stage2_val_cost_arr):
            stage2_train_cost_arr.append(train(stage1_model, stage2_model, dataset[0], stage2_loss, stage2_optimizer, stage2_scheduler, epoch))
            '''validation'''
            stage2_val_cost_arr.append(validation(stage1_model, stage2_model, dataset[1], stage2_loss, epoch))
            if epoch % 1 == 0:
                ''' test model for each n epochs'''
                test_cost, plot_img = test(stage1_model, stage2_model, dataset[2], stage2_loss, epoch, plot_scaled=False)
                stage2_test_cost_arr.append(test_cost)
                img_flag = True
            if epoch != 0:
                """ save model if train cost and val cost are lower than mean of previous epochs """
                # if train_cost_arr[-1] < train_cost_arr[-2] and val_cost_arr[-1] < val_cost_arr[-2]:
                print(np.min(stage2_train_cost_arr[:-1]), np.min(stage2_val_cost_arr[:-1]))
                print(stage2_train_cost_arr[-1], stage2_val_cost_arr[-1])
                if stage2_train_cost_arr[-1] < np.min(stage2_train_cost_arr[:-1]) and stage2_val_cost_arr[-1] < np.min(stage2_val_cost_arr[:-1]):
                    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    stage2_save_point.append(current_time)
                    stage2_best_model_path = model_save(stage2_train_cost_arr, stage2_val_cost_arr, stage2_model, stage2_save_point, model_name, dataset_name)
            if wandb_on:
                ''' wandb logging '''
                if epoch == 0:
                    wandb.init(project="VBPNet", entity="paperchae")

                wandb.log({"Stage2_train_cost": stage2_train_cost_arr[-1],
                           "Stage2_val_cost": stage2_val_cost_arr[-1],
                           "Stage2_test_cost": stage2_test_cost_arr[-1]}, step=epoch)
                # if epoch!=0:
                if img_flag:
                    wandb.log({"Stage2_Prediction": wandb.Image(plot_img)})
                    plot_img.close()
        else:
            print("model is not learning, stop training..")
            print("best model path :{}".format(stage2_best_model_path))
            break
        """ plot stage 2 Loss graph """
        t = np.array(range(len(stage2_train_cost_arr)))
        plt.title('Total epochs : {}'.format(len(t)))
        plt.plot(t, stage2_train_cost_arr, 'g-', label='Train Loss')
        plt.plot(t, stage2_val_cost_arr, 'b--', label='Validation Loss')
        plt.plot(t, stage2_test_cost_arr, 'r--', label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
