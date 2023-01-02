from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import vid2bp.utils.train_utils as tu


def train(model, dataset, loss, optimizer, scheduler, epoch, scaler=True):
    model.train()

    avg_cost_list = []
    for _ in range(len(loss)):
        avg_cost_list.append(0)

    with tqdm(dataset, desc='Train{}'.format(str(epoch)), total=len(dataset),
              leave=True) as train_epoch:
        for idx, (X_train, Y_train, d, s, size_class) in enumerate(train_epoch):
            optimizer.zero_grad()
            hypothesis = model(X_train, scaler=scaler)
            avg_cost_list, cost = tu.calc_losses(avg_cost_list, loss,
                                                 hypothesis, Y_train,
                                                 idx + 1)
            total_cost = np.sum(avg_cost_list)
            temp = {}
            for i in range(len(loss)):
                temp[(str(loss[i]))[:-2]] = (round(avg_cost_list[i], 3))
            train_epoch.set_postfix(losses=temp, tot=total_cost)
            cost.backward()
            optimizer.step()

        scheduler.step()
        # wandb.init(project="VBPNet", entity="paperchae")
        # wandb.log({'Train Loss': total_cost}, step=epoch)
        # wandb.log({'Train Loss': train_avg_cost,
        #            'Pearson Loss': neg_cost,
        #            'STFT Loss': stft_cost}, step=epoch)
        # wandb.log({"Train Loss": cost,
        #            "Train Negative Pearson Loss": neg_cost,  # },step=epoch)
        #            "Train Systolic Loss": s_cost,
        #            "Train Diastolic Loss": d_cost}, step=epoch)
    return total_cost.__float__()
