import json
import h5py

with open('/home/paperc/PycharmProjects/VBPNet/config/parameter.json') as f:
    json_data = json.load(f)
    param = json_data.get("parameters")
    root_path = json_data.get("parameters").get("root_path")  # mimic uci containing folder
    data_path = json_data.get("parameters").get("dataset_path")  # raw mimic uci selection
    orders = json_data.get("parameters").get("in_channels")
    sampling_rate = json_data.get("parameters").get("sampling_rate")


def selector(dataset, degree, samp_rate, cv=0):
    read_path = root_path + data_path[dataset][0]
    write_path = root_path + data_path[dataset][1]
    # TODO MIMIC DATA_AGGREGATOR NEEDS TO BE DONE
    if dataset == "mimic":
        import MIMICdataset
        print("mimic dataset selected")
        # train_abp, train_ple, data_len = MIMICdataset.data_aggregator(root_path=read_path, degree=degree,
        #                                                               train=train, percent=0.05,
        #                                                               samp_rate=samp_rate)  # 0.05 -> 2 patients
    else:  # uci
        import UCIdataset
        print("uci dataset selected")
        save_path = write_path + 'case(' + str(degree[-1]) + ')_' + \
                    str(int(param['chunk_size'] / sampling_rate['base']) * samp_rate)
        UCIdataset.data_aggregator2(read_path, save_path, degree[1], samp_rate, cv)
        # train_abp, train_ple, train_size, data_len = UCIdataset.data_aggregator(root_path=read_path, degree=degree,
        #                                                                         train=train, percent=0.75,
        #                                                                         samp_rate=samp_rate)  # 0.25 -> 3000 patients (total 12000 patient)

    # dset = h5py.File(write_path + "/AHAclass/case(" + str(order[-1]) + ")_len(" + str(data_len) + ")_"
    #                  + str(int(param["chunk_size"] / sampling_rate["base"]) * samp_rate)
    #                  + "_train(" + str(train) + ")_checker_5(crisis).hdf5", "w")
    # if len(train_ple) == len(train_abp):
    #     dset['ple'] = train_ple
    #     dset['abp'] = train_abp
    #     dset['size'] = train_size
    #     # dset['size'] = size_factor
    # else:
    #     print("length of ple and abp doesn't match")
    # dset.close()


order = orders["sixth"]
samp_rate = sampling_rate["60"]
selector(dataset="uci", degree=order, samp_rate=samp_rate, cv=1)
