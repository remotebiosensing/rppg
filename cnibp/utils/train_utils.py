import os
import json
import numpy as np

import h5py
from cnibp import loss

import torch
import torch.nn as nn
import torch.optim as optim


def get_model_parameter(model_name: str = 'BPNet'):
    with open('/home/paperc/PycharmProjects/Pytorch_rppgs/cnibp/configs/parameter.json') as f:
        j = json.load(f)
        params = j.get("models").get(model_name)
        out_channels = j.get("parameters").get("out_channels")
    learning_rate = params.get("learning_rate")
    weight_decay = params.get("weight_decay")
    gamma = params.get("gamma")
    return learning_rate, weight_decay, gamma, out_channels


def get_model(model_name: str, channel: int, device, stage: int = 1):
    lr, wd, ga, oc = get_model_parameter(model_name)
    if model_name == 'BPNet':
        if stage == 1:
            from cnibp.nets.bvp2abp import bvp2abp
            model = bvp2abp(in_channels=channel,
                            out_channels=oc,
                            target_samp_rate=60).to(device)
            # model_loss = [loss.NegPearsonLoss().to(device), loss.AmpLoss().to(device)]
            model_loss = [loss.NegPearsonLoss().to(device), loss.DBPLoss().to(device), loss.SBPLoss().to(device)]
            model_optim = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            model_scheduler = optim.lr_scheduler.ExponentialLR(model_optim, gamma=ga)
            # model_scheduler = optim.lr_scheduler.LambdaLR(optimizer=model_optim, lr_lambda=ga)
        else:
            from cnibp.nets.refactoring.ver2.SignalAmplificationModule import SignalAmplifier
            model = SignalAmplifier().to(device)
            model_loss = [loss.DBPLoss().to(device), loss.SBPLoss().to(device), loss.ScaleLoss().to(device)]
            model_optim = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            model_scheduler = optim.lr_scheduler.ExponentialLR(model_optim, gamma=ga)
    elif model_name == 'Unet':
        from cnibp.nets.unet import Unet
        model = Unet()
        model_loss = [nn.MSELoss()]
        model_optim = optim.Adam(model.parameters(), lr=lr)
        model_scheduler = None  # optim.lr_scheduler.ExponentialLR(model_optim, gamma=ga)
    else:
        print('not supported model name error')
        return
    return model, model_loss, model_optim, model_scheduler


def is_learning(cost_arr):
    # print('cost :', cost_arr)
    flag = True
    cnt = 0
    if len(cost_arr) > 5:
        last_five_cost_mean = np.mean(cost_arr[-5:])
        print('last five cost :', cost_arr[-5:])
        print('last five cost mean :', last_five_cost_mean)
        for c in reversed(cost_arr[-5:]):
            if c > last_five_cost_mean:
                cnt += 1
                print(c)
            if cnt > 3:
                print(cnt)
                flag = False
                break

    return flag


def get_avg_cost(avg_cost, current_cost, cnt):
    return (avg_cost * (cnt - 1) + current_cost.item()) / cnt


def calc_losses(avg_cost_list, loss_list, hypothesis, target, cnt):
    total_cost = torch.tensor(0.0).to('cuda:0')
    for idx, l in enumerate(loss_list):
        current_cost = l(hypothesis, target)
        total_cost += current_cost
        avg_cost_list.append(get_avg_cost(avg_cost_list[idx], current_cost, cnt))
        avg_cost_list[idx] = get_avg_cost(avg_cost_list[idx], current_cost, cnt)

    # cost = sum(cost_list)

    return avg_cost_list, total_cost


# def get_avg_cost(avg_cost, current_cost, cnt):
#     return (avg_cost * (cnt - 1) + current_cost.item()) / cnt


def model_save(train_cost_arr, val_cost_arr, model, save_point, model_name, dataset_name, channel, gender):
    print('\ncurrent train cost :', round(train_cost_arr[-1], 4), '/ prior_cost :', round(train_cost_arr[-2], 4),
          ' >> trained :', round(train_cost_arr[-2] - train_cost_arr[-1], 4))
    print('current val cost :', round(val_cost_arr[-1], 4), '/ prior_cost :', round(val_cost_arr[-2], 4),
          ' >> trained :', round(val_cost_arr[-2] - val_cost_arr[-1], 4))
    save_path = "/home/paperc/PycharmProjects/Pytorch_rppgs/cnibp/weights/" + \
                '{}_{}_{}_{}_{}.pt'.format(model_name, dataset_name, save_point[-1], channel, gender)
    print('saving model :', save_path)
    # torch.save(model.state_dict(), save_path)
    torch.save(model, save_path)
    try:
        prior_path = "/home/paperc/PycharmProjects/Pytorch_rppgs/cnibp/weights/" + \
                     '{}_{}_{}_{}_{}.pt'.format(model_name, dataset_name, save_point[-2], channel, gender)
        # print(prior_path)
        os.remove(prior_path)
        print('removed prior model :', prior_path)
        return save_path
    except:
        print('failed to remove prior model')
        return save_path






def train_test_shuffler(tr_ple, te_ple, tr_abp, te_abp, tr_size, te_size):
    import random
    print('shuffler called')
    with open('/home/paperc/PycharmProjects/VBPNet/configs/parameter.json') as f:
        json_data = json.load(f)
        root_path = json_data.get("parameters").get("root_path")  # mimic uci containing folder
        data_path = json_data.get("parameters").get("dataset_path")  # raw mimic uci selection
        orders = json_data.get("parameters").get("in_channels")

    order = orders["sixth"]
    write_path = root_path + data_path[dataset][1]

    ple_total = np.concatenate((tr_ple, te_ple), axis=0)
    abp_total = np.concatenate((tr_abp, te_abp), axis=0)
    size_total = np.concatenate((tr_size, te_size), axis=0)
    total_data = [[p, a, s] for p, a, s in zip(ple_total, abp_total, size_total)]
    random.shuffle(total_data)
    total_len = len(total_data)
    train_len = int(total_len * 0.8)
    shuffled_ple = np.squeeze([n[0] for n in total_data])
    shuffled_abp = np.squeeze([n[1] for n in total_data])
    shuffled_size = np.squeeze([n[2] for n in total_data])
    shuffled_tr_ple, shuffled_te_ple = np.split(shuffled_ple, [train_len])
    shuffled_tr_abp, shuffled_te_abp = np.split(shuffled_abp, [train_len])
    shuffled_tr_size, shuffled_te_size = np.split(shuffled_size, [train_len])
    print("train size :", len(shuffled_tr_ple), "test size :", len(shuffled_te_size))

    shuffled_train_dset = h5py.File(write_path + "/shuffled/case(" + str(order[-1]) + ")_"
                                    + str(int(param["chunk_size"] / sampling_rate["base"]) * samp_rate)
                                    + "_shuffled_train.hdf5", "w")
    shuffled_train_dset['ple'] = shuffled_tr_ple
    shuffled_train_dset['abp'] = shuffled_tr_abp
    shuffled_train_dset['size'] = shuffled_tr_size
    shuffled_train_dset.close()

    shuffled_test_dset = h5py.File(write_path + "/shuffled/case(" + str(order[-1]) + ")_"
                                   + str(int(param["chunk_size"] / sampling_rate["base"]) * samp_rate)
                                   + "_shuffled_test.hdf5", "w")
    shuffled_test_dset['ple'] = shuffled_te_ple
    shuffled_test_dset['abp'] = shuffled_te_abp
    shuffled_test_dset['size'] = shuffled_te_size
    shuffled_test_dset.close()

    distribution_checker(shuffled_tr_size, shuffled_te_size)


def data_shuffler(model_name, save_path, input_data, cv):
    import random
    print('data shuffler called')
    ple = input_data[0]
    abp = input_data[1]
    if model_name == "BPNet":
        size = input_data[2]
        total_data = [[p, a, s] for p, a, s in zip(ple, abp, size)]
        random.shuffle(total_data)
        total_len = len(total_data)
        if cv == 1:
            train_val_len = int(total_len * 0.85)
        else:
            train_val_len = int(total_len * 0.8)
        shuffled_ple = np.squeeze([n[0] for n in total_data])
        shuffled_abp = np.squeeze([n[1] for n in total_data])
        shuffled_size = np.squeeze([n[2] for n in total_data])
        print('***********\nnp.shape(shuffled_ple) :', np.shape(shuffled_ple))
        shuffled_tr_ple, shuffled_te_ple = np.split(shuffled_ple, [train_val_len])
        shuffled_tr_abp, shuffled_te_abp = np.split(shuffled_abp, [train_val_len])
        shuffled_tr_size, shuffled_te_size = np.split(shuffled_size, [train_val_len])
        print('np.shape(shuffled_tr_ple) :', np.shape(shuffled_tr_ple))
        print('np.shape(shuffled_te_ple) :', np.shape(shuffled_te_ple))
        print('data shuffle for BPNet done')
        test_dset = h5py.File(save_path + "_test.hdf5", "w")
        test_dset['ple'] = shuffled_te_ple
        test_dset['abp'] = shuffled_te_abp
        test_dset['size'] = shuffled_te_size
        print('test data for BPNet saved')

        if cv >= 1:
            print('cross validation :', cv)
            k_fold_cv(model_name, save_path, [shuffled_tr_ple, shuffled_tr_abp, shuffled_tr_size], cv)
            print('train data for BPNet saved')

        else:
            print('not supported fold number')
    elif model_name == "Unet":
        total_data = [[p, a] for p, a in zip(ple, abp)]
        random.shuffle(total_data)
        total_len = len(total_data)
        train_val_len = int(total_len * 0.8)
        shuffled_ple = np.squeeze([n[0] for n in total_data])
        shuffled_abp = np.squeeze([n[1] for n in total_data])
        print('***********\nnp.shape(shuffled_ple) :', np.shape(shuffled_ple))
        shuffled_tr_ple, shuffled_te_ple = np.split(shuffled_ple, [train_val_len])
        shuffled_tr_abp, shuffled_te_abp = np.split(shuffled_abp, [train_val_len])
        print('np.shape(shuffled_tr_ple) :', np.shape(shuffled_tr_ple))
        print('np.shape(shuffled_te_ple) :', np.shape(shuffled_te_ple))

        print('data shuffle for Unet done')
        test_dset = h5py.File(save_path + "_test.hdf5", "w")
        test_dset['ple'] = shuffled_te_ple
        test_dset['abp'] = shuffled_te_abp
        print('test data for Unet saved')

        if cv >= 1:
            print('cross validation :', cv)
            k_fold_cv(model_name, save_path, [shuffled_tr_ple, shuffled_tr_abp], cv)
            print('train data for Unet saved')

        else:
            print('not supported fold number')


def k_fold_cv(model_name, save_path, train_data, cv_n):
    if model_name == "BPNet":
        ple = train_data[0]
        abp = train_data[1]
        size = train_data[2]

        if len(ple) % cv_n != 0:
            ple = ple[:len(ple) - len(ple) % cv_n]
            abp = abp[:len(abp) - len(abp) % cv_n]
            size = size[:len(size) - len(size) % cv_n]

        ple_total, abp_total, size_total = [], [], []
        ple_validation_total, abp_validation_total, size_validation_total = [], [], []

        if cv_n == 1:
            # normal train validation >> divide train : 80% / validation : 20%
            # ple_total, abp_total, size_total = ple, abp, size
            ple_total, ple_validation_total = np.split(ple, [int(len(ple) * 0.8)])
            abp_total, abp_validation_total = np.split(abp, [int(len(abp) * 0.8)])
            size_total, size_validation_total = np.split(size, [int(len(size) * 0.8)])

        else:
            # n fold cross validation >> divide data into n folds
            ple_folds = np.split(ple, cv_n)
            abp_folds = np.split(abp, cv_n)
            size_folds = np.split(size, cv_n)
            for idx, (ple_fold, abp_fold, size_fold) in enumerate(zip(ple_folds, abp_folds, size_folds)):
                ple_temp = []
                abp_temp = []
                size_temp = []
                for i in range(cv_n):
                    if idx != i:
                        ple_temp.append(ple_folds[i])
                        abp_temp.append(abp_folds[i])
                        size_temp.append(size_folds[i])
                        # print(i)
                ple_total.append(np.reshape(ple_temp, (-1, 3, chunk_size)))
                abp_total.append(np.reshape(abp_temp, (-1, chunk_size)))
                size_total.append(np.reshape(size_temp, (-1, 3)))

                ple_validation_total.append(ple_fold)
                abp_validation_total.append(abp_fold)
                size_validation_total.append(size_fold)

        dset = h5py.File(save_path + '_train(cv' + str(cv_n) + ').hdf5', "w")
        t_dset = dset.create_group('train')
        t_dset.attrs['desc'] = "k-fold cv train dataset"
        t_p = t_dset.create_group('ple')
        t_a = t_dset.create_group('abp')
        t_s = t_dset.create_group('size')

        v_dset = dset.create_group('validation')
        v_dset.attrs['desc'] = "k-fold cv validation dataset"
        v_p = v_dset.create_group('ple')
        v_a = v_dset.create_group('abp')
        v_s = v_dset.create_group('size')
        if cv_n == 1:
            t_p.create_dataset(name=str(0), data=ple_total)
            t_a.create_dataset(name=str(0), data=abp_total)
            t_s.create_dataset(name=str(0), data=size_total)
            v_p.create_dataset(name=str(0), data=ple_validation_total)
            v_a.create_dataset(name=str(0), data=abp_validation_total)
            v_s.create_dataset(name=str(0), data=size_validation_total)
        else:
            for n in range(cv_n):
                t_p.create_dataset(name=str(n), data=ple_total[n])
                t_a.create_dataset(name=str(n), data=abp_total[n])
                t_s.create_dataset(name=str(n), data=size_total[n])
                v_p.create_dataset(name=str(n), data=ple_validation_total[n])
                v_a.create_dataset(name=str(n), data=abp_validation_total[n])
                v_s.create_dataset(name=str(n), data=size_validation_total[n])

        print('t_dset.keys() :', t_dset.keys())
        print('t_p.keys() :', t_p.keys())
        print('done')
        hdf_handler.cv_dataset_reader(save_path + '_train(cv' + str(cv_n) + ').hdf5')
    elif model_name == "Unet":
        ple = train_data[0]
        abp = train_data[1]
        if len(ple) % cv_n != 0:
            ple = ple[:len(ple) - len(ple) % cv_n]
            abp = abp[:len(abp) - len(abp) % cv_n]

        if cv_n == 1:
            ple_total, ple_validation_total = np.split(ple, [int(len(ple) * 0.8)])
            abp_total, abp_validation_total = np.split(abp, [int(len(abp) * 0.8)])

        else:  # cv_n > 1
            ple_folds = np.split(ple, cv_n)
            abp_folds = np.split(abp, cv_n)
            ple_total, abp_total = [], []
            ple_validation_total, abp_validation_total = [], []

            for idx, (ple_fold, abp_fold) in enumerate(zip(ple_folds, abp_folds)):
                ple_temp = []
                abp_temp = []
                for i in range(cv_n):
                    if idx != i:
                        ple_temp.append(ple_folds[i])
                        abp_temp.append(abp_folds[i])
                        # print(i)
                ple_total.append(np.reshape(ple_temp, (-1, chunk_size)))
                abp_total.append(np.reshape(abp_temp, (-1, chunk_size)))

                ple_validation_total.append(ple_fold)
                abp_validation_total.append(abp_fold)

        dset = h5py.File(save_path + '_train(cv' + str(cv_n) + ').hdf5', "w")
        t_dset = dset.create_group('train')
        t_dset.attrs['desc'] = "k-fold cv train dataset"
        t_p = t_dset.create_group('ple')
        t_a = t_dset.create_group('abp')

        v_dset = dset.create_group('validation')
        v_dset.attrs['desc'] = "k-fold cv validation dataset"
        v_p = v_dset.create_group('ple')
        v_a = v_dset.create_group('abp')

        for n in range(cv_n):
            t_p.create_dataset(name=str(n), data=ple_total[n])
            t_a.create_dataset(name=str(n), data=abp_total[n])
            v_p.create_dataset(name=str(n), data=ple_validation_total[n])
            v_a.create_dataset(name=str(n), data=abp_validation_total[n])

        print('t_dset.keys() :', t_dset.keys())
        print('t_p.keys() :', t_p.keys())
        print('done')
        # hdf_handler.cv_dataset_reader(save_path + '_train(cv' + str(cv_n) + ').hdf5')
    else:
        print('not supported model in cv_dataset_maker')
        return
