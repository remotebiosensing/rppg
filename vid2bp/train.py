import wandb
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def train(model, dataset, loss, optimizer, scheduler, epoch, scaler=True):
    model.train()
    train_avg_cost = 0
    train_cost_sum = 0
    with tqdm(dataset, desc='Train{}'.format(str(epoch)), total=len(dataset),
              leave=True) as train_epoch:
        # avg_cost = 0
        for idx, (X_train, Y_train, d, s, m) in enumerate(train_epoch):
            optimizer.zero_grad()
            hypothesis = model(X_train, scaler=scaler)

            '''Negative Pearson Loss'''
            neg_cost = loss[0](hypothesis, Y_train)
            # neg_cost = 0
            ''' STFT Loss '''
            stft_cost = loss[1](hypothesis, Y_train)
            '''DBP Loss'''
            # d_cost = loss[1](pred_d, d)
            '''SBP Loss'''
            # s_cost = loss[2](pred_s, s)
            '''FFT Loss'''
            # fft_cost = loss_fft(hypothesis, Y_train)

            '''Total Loss'''
            cost = neg_cost + stft_cost
            # cost = fft_cost
            cost.backward()
            optimizer.step()

            if not np.isnan(cost.__float__()):
                train_cost_sum += cost.__float__()
                train_avg_cost = train_cost_sum / (idx + 1)
                train_epoch.set_postfix(n=neg_cost.__float__(), s=stft_cost.__float__(), loss=train_avg_cost)
            else:
                print('nan error')
                continue
        scheduler.step()
        wandb.log({'Train Loss': train_avg_cost}, step=epoch)
        # wandb.log({'Train Loss': train_avg_cost,
        #            'Pearson Loss': neg_cost,
        #            'STFT Loss': stft_cost}, step=epoch)
        # wandb.log({"Train Loss": cost,
        #            "Train Negative Pearson Loss": neg_cost,  # },step=epoch)
        #            "Train Systolic Loss": s_cost,
        #            "Train Diastolic Loss": d_cost}, step=epoch)
    return train_avg_cost.__float__()
