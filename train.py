import wandb
from tqdm import tqdm
import torch
from nets.loss import loss
import json
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from test import test

with open('config/parameter.json') as f:
    hyper_param = json.load(f).get("hyper_parameters")


def train(model, device, train_loader, test_loader, epochs):
    if torch.cuda.is_available():
        model = model.to(device)
        loss_neg = loss.NegPearsonLoss().to(device)
        loss_d = loss.dbpLoss().to(device)
        loss_s = loss.sbpLoss().to(device)

    else:
        print("Use Warning : Please load model on cuda! (Loaded on CPU)")
        model = model.to('cpu')
        loss_neg = loss.NegPearsonLoss().to('cpu')
        loss_d = loss.dbpLoss().to('cpu')
        loss_s = loss.sbpLoss().to('cpu')

    """optimizer"""
    optimizer = optim.AdamW(model.parameters(), lr=hyper_param["learning_rate"],
                            weight_decay=hyper_param["weight_decay"])
    """scheduler"""
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=hyper_param["gamma"])

    print('batchN :', train_loader.__len__())

    costarr = []
    for epoch in range(epochs):
        avg_cost = 0
        cost_sum = 0
        if epoch % 10 == 0:
            test(model=model, test_loader=test_loader, loss_n=loss_neg, loss_d=loss_d, loss_s=loss_s, idx=epoch)

        with tqdm(train_loader, desc='Train', total=len(train_loader), leave=True) as train_epochs:
            idx = 0
            for X_train, Y_train, d, s in train_epochs:
                idx += 1
                hypothesis = torch.squeeze(model(X_train)[0])
                pred_d = torch.squeeze(model(X_train)[1])
                pred_s = torch.squeeze(model(X_train)[2])
                optimizer.zero_grad()

                '''Negative Pearson Loss'''
                neg_cost = loss_neg(hypothesis, Y_train)
                '''DBP Loss'''
                d_cost = loss_d(pred_d, d)
                '''SBP Loss'''
                s_cost = loss_s(pred_s, s)
                '''Amplitude Loss'''
                '''MBP Loss'''
                # m_cost = loss2(pred_m, m)
                # TODO test while training every 10 epoch + wandb graph


                '''Total Loss'''
                cost = neg_cost + d_cost + s_cost  # + d_cost
                cost.backward()
                optimizer.step()

                # avg_cost += cost / train_loader.__len__()
                cost_sum += cost
                avg_cost = cost_sum / idx
                train_epochs.set_postfix(loss=avg_cost.item())
                wandb.log({"Train Loss": cost,
                           "Train Negative Pearson Loss": neg_cost,  # },step=epoch)
                           "Train Systolic Loss": s_cost,
                           "Train Diastolic Loss": d_cost}, step=epoch)

            scheduler.step()
            costarr.append(avg_cost.__float__())
        if epoch % 10 == 0:
            test(model=model, test_loader=test_loader, loss_n=loss_neg, loss_d=loss_d, loss_s=loss_s, idx=epoch)

    print('cost :', costarr[-1])

    t_val = np.array(range(len(costarr)))
    plt.plot(t_val, costarr)
    plt.title('NegPearsonLoss * MAELoss')
    plt.show()
