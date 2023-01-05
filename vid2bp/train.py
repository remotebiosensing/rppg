from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import vid2bp.utils.train_utils as tu
from vid2bp.nets.loss.loss import SelfScaler


def train(model, dataset, abp_losses, optimizer, scheduler, epoch, scaler=True):
    model.train()
    scale_loss = SelfScaler().to('cuda:0')

    avg_cost_list = []
    for _ in range(len(abp_losses)):
        avg_cost_list.append(0)

    with tqdm(dataset, desc='Train{}'.format(str(epoch)), total=len(dataset),
              leave=True) as train_epoch:
        for idx, (X_train, Y_train, d, s, size_class) in enumerate(train_epoch):
            optimizer.zero_grad()
            hypothesis, scaled_ple = model(X_train, scaler=scaler)
            avg_cost_list, cost = tu.calc_losses(avg_cost_list, abp_losses,
                                                 hypothesis, Y_train,
                                                 idx + 1)

            ple_cost = scale_loss(scaled_ple, X_train)
            total_cost = np.sum(avg_cost_list) + ple_cost.__float__()

            postfix_dict = {}
            for i in range(len(abp_losses)):
                postfix_dict[(str(abp_losses[i]))[:-2]] = (round(avg_cost_list[i], 3))
            postfix_dict['scale_variance'] = round(ple_cost.__float__(), 3)
            train_epoch.set_postfix(losses=postfix_dict, tot=total_cost)
            (cost + ple_cost).backward()
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
