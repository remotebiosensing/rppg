import os
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import json
import wandb
import vid2bp.preprocessing.utils.signal_utils as su
import vid2bp.postprocessing.post_signal_utils as psu
import numpy as np
import vid2bp.utils.train_utils as tu


# with open('/home/paperc/PycharmProjects/Pytorch_rppgs/vid2bp/config/parameter.json') as f:
#     json_data = json.load(f)
#     param = json_data.get("parameters")
#     channels = json_data.get("parameters").get("in_channels")
#     sampling_rate = json_data.get("parameters").get("sampling_rate")
#
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.manual_seed(125)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(125)
# else:
#     print("cuda not available")


def validation(model, dataset, loss, epoch, scaler=True):
    model.eval()

    avg_cost_list = []
    for _ in range(len(loss)):
        avg_cost_list.append(0)

    with tqdm(dataset, desc='Validation{}'.format(str(epoch)), total=len(dataset), leave=True) as valid_epoch:
        with torch.no_grad():
            for idx, (X_val, Y_val, dia, sys, size_class) in enumerate(valid_epoch):
                hypothesis = model(X_val, scaler=scaler)
                avg_cost_list, cost = tu.calc_losses(avg_cost_list, loss,
                                                     hypothesis, Y_val,
                                                     idx + 1)
                total_cost = np.sum(avg_cost_list)
                temp = {}
                for i in range(len(loss)):
                    temp[(str(loss[i]))[:-2]] = (round(avg_cost_list[i], 3))

                valid_epoch.set_postfix(losses=temp, tot=total_cost)
            # wandb.init(project="VBPNet", entity="paperchae")
            # wandb.log({"Valid Loss": total_cost}, step=epoch)
            # wandb.log({"Valid Loss": valid_avg_cost,
            #            'Valid Pearson Loss': neg_cost,
            #            'STFT Loss': stft_cost}, step=epoch)
        return total_cost.__float__()
