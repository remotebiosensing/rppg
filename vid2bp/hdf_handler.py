import h5py
import numpy as np

cv = 5
dpath = '/home/paperc/PycharmProjects/VBPNet/dataset/BPNet_uci/case(P+V+A)_360_train(cv' + str(cv) + ').hdf5'


def cv_dataset_reader(path):
    with h5py.File(path, 'r') as f:
        print('dset.keys() :', f.keys())

        train_group_key = list(f.keys())[0]
        validation_group_key = list(f.keys())[1]

        print(f[train_group_key])
        print(f[train_group_key].keys())
        print(f[validation_group_key])
        print(f[validation_group_key].keys())
        # val = list(f[validation_group_key].keys())
        # print(f[val])
        # print(type(f[val]))
        group_list = ['train', 'validation']
        sub_group_list = ['ple', 'abp', 'size']
        for g in group_list:
            for s in sub_group_list:
                for l in range(len(f[g + '/' + s])):
                    # if l == 0:
                    print(g, s, l)
                    print(np.shape(f[g + '/' + s + '/' + str(l)]))
        print(len(f['validation/abp']))
        # ds_arr = f[validation_group_key][()]
        # print(np.shape(ds_arr))


# cv_dataset_reader(dpath)
'''
User manual
k-fold cross validation datasets are made,
to use train and val dataset, simply call same index train[ple, abp, size] and val[ple, abp, size] 
'''
