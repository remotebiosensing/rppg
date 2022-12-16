import h5py
import numpy as np


def cv_dataset_reader(path):
    with h5py.File(path, 'r') as f:
        print('dset.keys() :', f.keys())
        if 'Unet' in path:

            train_group_key = list(f.keys())[0]
            print(f[train_group_key])
            print(f[train_group_key].keys())
            group_list = ['train']
            sub_group_list = ['ple', 'abp']

        else:
            train_group_key = list(f.keys())[0]
            validation_group_key = list(f.keys())[1]
            print(f[train_group_key])
            print(f[train_group_key].keys())
            print(f[validation_group_key])
            print(f[validation_group_key].keys())
            group_list = ['train', 'validation']
            sub_group_list = ['ple', 'abp', 'size']

        for g in group_list:
            for s in sub_group_list:
                for l in range(len(f[g + '/' + s])):
                    # if l == 0:
                    print(g, s, l)
                    print(np.shape(f[g + '/' + s + '/' + str(l)]))
        print('len', np.shape(np.squeeze(f['train/ple/0'])))
        # ds_arr = f[validation_group_key][()]
        # print(np.shape(ds_arr))


cv = 1
dpath = '/home/paperc/PycharmProjects/dataset/Unet_uci/case(P)_360_train(cv' + str(cv) + ')256.hdf5'

# cv_dataset_reader(dpath)
'''
User manual
k-fold cross validation datasets are made,
to use train and val dataset, simply call same index train[ple, abp, size] and val[ple, abp, size] 
'''
