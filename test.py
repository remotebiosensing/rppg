import os
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import json
import wandb

with open('/home/paperc/PycharmProjects/VBPNet/config/parameter.json') as f:
    json_data = json.load(f)
    param = json_data.get("parameters")
    channels = json_data.get("parameters").get("in_channels")
    sampling_rate = json_data.get("parameters").get("sampling_rate")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(125)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(125)
else:
    print("cuda not available")


def test(model, test_loader, loss_n, loss_d, loss_s, idx):
    dataiter = iter(test_loader)
    # seq, labels = dataiter.__next__()
    model.eval()
    plot_flag = True
    with tqdm(test_loader, desc='Test', total=len(test_loader), leave=True) as test_epoch:
        with torch.no_grad():
            for X_test, Y_test, d, s in test_epoch:

                hypothesis = torch.squeeze(model(X_test)[0])
                pred_d = torch.squeeze(model(X_test)[1])
                pred_s = torch.squeeze(model(X_test)[2])

                '''Negative Pearson Loss'''
                neg_cost = loss_n(hypothesis, Y_test)
                '''DBP Loss'''
                d_cost = loss_d(pred_d, d)
                '''SBP Loss'''
                s_cost = loss_s(pred_s, s)
                ''' Total Loss'''
                cost = neg_cost + d_cost + s_cost
                test_epoch.set_postfix(loss=cost.item())
                wandb.log({"Test Loss": cost,
                           "Test Negative Pearson Loss": neg_cost,
                           "Test Systolic Loss": s_cost,
                           "Test Diastolic Loss": d_cost}, step=idx)
                if plot_flag:
                    plot_flag = False
                    h = hypothesis[0].cpu().detach()
                    y = Y_test[0].cpu().detach()
                    plt.subplot(2, 1, 1)
                    plt.plot(y)
                    plt.title("Target (epoch :" + str(idx) + ")")
                    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.70)

                    plt.subplot(2, 1, 2)
                    plt.plot(h)
                    plt.title("Prediction (epoch :" + str(idx) + ")")
                    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.70)
                    wandb.log({"Prediction": wandb.Image(plt)})
                    plt.show()

