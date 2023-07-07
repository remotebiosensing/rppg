import math
import torch
import wandb
from tqdm import tqdm
from rppg.utils.funcs import (get_hr, MAE, RMSE, MAPE, corr, IrrelevantPowerRatio)
import numpy as np
import os


def run(model, sweep, optimizer, lr_sch, criterion, cfg, dataloaders, wandb_flag):
    log = True
    best_loss = 100000
    val_loss = 0
    eval_flag = False
    save_dir = cfg.model_save_path + cfg.fit.model + "/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    test_result = []
    if cfg.fit.train_flag:
        for epoch in range(cfg.fit.train.epochs):
            train_fn(epoch, model, optimizer, lr_sch, criterion, dataloaders[0], wandb_flag)
            val_loss = val_fn(epoch, model, criterion, dataloaders[1], wandb_flag)
            if best_loss > val_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), save_dir +
                           "train" + cfg.fit.train.dataset +
                           "_test" + cfg.fit.test.dataset +
                           "_imgsize" + str(cfg.fit.img_size) +
                           ".pt")
                eval_flag = True
            if log:
                if cfg.fit.eval_flag and (eval_flag or (epoch + 1) % cfg.fit.eval_interval == 0):
                    test_fn(epoch, model, dataloaders[2], cfg.fit.model, cal_type=cfg.fit.test.cal_type,
                            metrics=cfg.fit.test.metric, eval_time_length=10,
                            wandb_flag=wandb_flag)
                eval_flag = False
        if not sweep:
            test_result = test_fn(0, model, dataloaders[2], cfg.fit.model, cal_type=cfg.fit.test.cal_type,
                                  metrics=cfg.fit.test.metric, eval_time_length=10,
                                  wandb_flag=wandb_flag)
        else:
            for et in cfg.fit.test.eval_time_length:
                test_result.append(test_fn(0, model, dataloaders[2], cfg.fit.model, cal_type=cfg.fit.test.cal_type,
                                           metrics=cfg.fit.test.metric, eval_time_length=et,
                                           wandb_flag=wandb_flag))
    else:
        # model = torch.load()
        if not sweep:
            test_result.append(test_fn(0, model, dataloaders[0], cfg.fit.model, cal_type=cfg.fit.test.cal_type,
                                       metrics=cfg.fit.test.metric, eval_time_length=cfg.fit.test.eval_time_length,
                                       wandb_flag=wandb_flag))
        else:
            for et in cfg.fit.test.eval_time_length:
                test_result.append(test_fn(0, model, dataloaders[0], cfg.fit.model, cal_type=cfg.fit.test.cal_type,
                                           metrics=cfg.fit.test.metric, eval_time_length=et,
                                           wandb_flag=wandb_flag))

    return test_result


def train_fn(epoch, model, optimizer, lr_sch, criterion, dataloaders, wandb_flag: bool = True):
    # TODO : Implement multiple loss
    step = "Train"

    with tqdm(dataloaders, desc=step, total=len(dataloaders)) as tepoch:
        model.train()
        running_loss = 0.0

        for te in tepoch:
            if model.__module__.split('.')[-1] == 'PhysFormer':
                inputs, target, hr = te
            else:
                inputs, target = te
            optimizer.zero_grad()
            tepoch.set_description(step + "%d" % epoch)
            outputs = model(inputs)
            if model.__module__.split('.')[-1] == 'PhysFormer':
                # temporal_loss = 0.0
                # ce_loss = 0.0
                # ld_loss = 0.0
                # batch_size = inputs.size()[0]
                # temporal_cost_function = NegPearsonLoss()
                # kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
                # temporal_loss = temporal_cost_function(outputs, target)
                #
                # fs, std = 30, 1.0
                # bpm_range = torch.arange(40, 180, dtype=torch.float32).cuda()
                # init_alpha, init_beta = 0.1, 1.0
                # alpha_exp, beta_exp = 0.5, 5.0
                # if epoch > 25:
                #     alpha = 0.05
                #     beta = 5.0
                # else:
                #     alpha = init_alpha * math.pow(alpha_exp, epoch / 25.0)
                #     beta = init_beta * math.pow(beta_exp, epoch / 25.0)
                # n = outputs.size()[-1]
                # unit_per_hz = n / fs
                # feasible_bpm = bpm_range / 60.0
                # k = (feasible_bpm / unit_per_hz).type(torch.FloatTensor).cuda().view(1, -1, 1)
                # two_pi_n_over_N = (Variable(2 * math.pi * torch.arange(0, n, dtype=torch.float32),
                #                             requires_grad=True) / n).cuda().view(1, 1, -1)
                # hanning = (Variable(torch.from_numpy(np.hanning(n)).type(torch.FloatTensor),
                #                     requires_grad=True).view(1, -1)).cuda()
                # hanning_rppg = (outputs * hanning).view(batch_size, 1, -1)  # (batch, 1, time length)
                # complex_absolute = torch.sum(hanning_rppg * torch.sin(k * two_pi_n_over_N), dim=-1) ** 2 + \
                #                    torch.sum(hanning_rppg * torch.cos(k * two_pi_n_over_N), dim=-1) ** 2
                # complex_absolute = (1.0 / complex_absolute.sum(keepdims=True, dim=-1)) * complex_absolute
                #
                # for b in range(batch_size):
                #     # ce loss
                #     ce_loss += F.cross_entropy(complex_absolute[b].view(1, -1), hr[b].view(1).type(torch.long) - 40)
                #     # ld loss
                #     target_distribution = []
                #     for i in range(140):
                #         target_distribution.append(math.exp(-(i - int(hr[b])) ** 2 / (2 * std ** 2)) /
                #                                    (math.sqrt(2 * math.pi) * std))
                #     target_distribution = torch.Tensor([i if i > 1e-15 else 1e-15 for i in target_distribution]).cuda()
                #     frequency_distribution = F.log_softmax(complex_absolute[b])
                #     ld_loss += kl_loss(frequency_distribution, target_distribution)
                # ce_loss /= batch_size
                # ld_loss /= batch_size
                #
                # loss = alpha * temporal_loss + beta * (ce_loss + ld_loss)
                loss = criterion(epoch, outputs, target, hr)
            else:
                loss = criterion(outputs, target)

            if ~torch.isfinite(loss):
                continue
            loss.requires_grad_(True)
            loss.backward(retain_graph=True)
            running_loss += loss.item()

            optimizer.step()
            if lr_sch is not None:
                lr_sch.step()

            tepoch.set_postfix({'': 'loss : %.4f | ' % (running_loss / tepoch.__len__())})

        if wandb_flag:
            wandb.log({"Train" + "_loss": running_loss / tepoch.__len__(),
                       },
                      step=epoch)


def val_fn(epoch, model, criterion, dataloaders, wandb_flag: bool = True):
    # TODO : Implement multiple loss
    # TODO : Implement save model function
    step = "Val"

    with tqdm(dataloaders, desc=step, total=len(dataloaders)) as tepoch:
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for te in tepoch:
                if model.__module__.split('.')[-1] == 'PhysFormer':
                    inputs, target, hr = te
                else:
                    inputs, target = te
                tepoch.set_description(step + "%d" % epoch)
                outputs = model(inputs)
                if model.__module__.split('.')[-1] == 'PhysFormer':

                    loss = criterion(epoch, outputs, target, hr)
                else:
                    loss = criterion(outputs, target)
                if ~torch.isfinite(loss):
                    continue
                running_loss += loss.item()
                tepoch.set_postfix({'': 'loss : %.4f |' % (running_loss / tepoch.__len__())})

            if wandb_flag:
                wandb.log({step + "_loss": running_loss / tepoch.__len__()},
                          step=epoch)

        return running_loss / tepoch.__len__()


def test_fn(epoch, model, dataloaders, model_name, cal_type, metrics, eval_time_length, wandb_flag: bool = True):
    # To evaluate a model by subject, you can use the meta option
    step = "Test"

    if model_name in ["DeepPhys", "TSCAN", "MTTS", "BigSmall", "EfficientPhys"]:
        model_type = 'DIFF'
    else:
        model_type = 'CONT'

    model.eval()

    p = []
    t = []

    fs = 30
    time = eval_time_length

    interval = fs * time

    for dataloader in dataloaders:
        with tqdm(dataloader, desc=step, total=len(dataloader), disable=True) as tepoch:
            _pred = []
            _target = []
            for te in tepoch:
                if model.__module__.split('.')[-1] == 'PhysFormer':
                    inputs, target, _ = te
                else:
                    inputs, target = te
                _pred.extend(np.reshape(model(inputs).cpu().detach().numpy(), (-1,)))
                _target.extend(np.reshape(target.cpu().detach().numpy(), (-1,)))

            remind = len(_pred) % interval
            if remind > 0:
                _pred = _pred[:-remind]
                _target = _target[:-remind]
        p.extend(np.reshape(np.reshape(np.asarray(_pred), -1), (-1, interval)))
        t.extend(np.reshape(np.reshape(np.asarray(_target), -1), (-1, interval)))
    p = np.asarray(p)
    t = np.asarray(t)

    hr_pred, hr_target = get_hr(p, t, model_type=model_type, cal_type=cal_type)
    hr_pred = np.asarray(hr_pred)
    hr_target = np.asarray(hr_target)

    test_result = []
    if "MAE" in metrics:
        test_result.append(round(MAE(hr_pred, hr_target), 3))
        print("MAE", MAE(hr_pred, hr_target))
    if "RMSE" in metrics:
        test_result.append(round(RMSE(hr_pred, hr_target), 3))
        print("RMSE", RMSE(hr_pred, hr_target))
    if "MAPE" in metrics:
        test_result.append(round(MAPE(hr_pred, hr_target), 3))
        print("MAPE", MAPE(hr_pred, hr_target))
    if "Pearson" in metrics:
        test_result.append(round(corr(hr_pred, hr_target)[0][1], 3))
        print("Pearson", corr(hr_pred, hr_target))
    return test_result


def find_lr(model, train_loader, optimizer, criterion, init_value=1e-8, final_value=10., beta=0.98):
    num = len(train_loader) - 1
    mult = (final_value / init_value) ** (1 / num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    losses = []
    log_lrs = []
    for batch_num, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # loss = loss[0]*10ㅌ+ loss[1]*5# + loss[2] #+ loss[3]
        # Compute the smoothed loss
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** (batch_num + 1))
        # Record the best loss
        if smoothed_loss < best_loss or batch_num == 0:
            best_loss = smoothed_loss
        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            break
        # Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        loss.requires_grad_(True)
        # Do the backward pass and update the weights
        loss.backward()
        optimizer.step()
        # Update the learning rate
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
    return log_lrs, losses

# calculate Heart Rate from PPG signal
