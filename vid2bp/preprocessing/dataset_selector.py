import json
import vid2bp.preprocessing.utils.math_module as mm
import vid2bp.preprocessing.utils.train_utils as tu

with open('/home/paperc/PycharmProjects/Pytorch_rppgs/vid2bp/config/parameter.json') as f:
    json_data = json.load(f)
    param = json_data.get("parameters")
    root_path = json_data.get("parameters").get("root_path")  # mimic uci containing folder
    data_path = json_data.get("parameters").get("dataset_path")  # raw mimic uci selection
    orders = json_data.get("parameters").get("in_channels")
    sampling_rate = json_data.get("parameters").get("sampling_rate")
    models = json_data.get("parameters").get("models")
    chunk_size = json_data.get("parameters").get("chunk_size")


def selector(model_name, dataset_name, degree, samp_rate, cv=1):
    print('< Blood Pressure Estimation dataset selector >')
    read_path = root_path + data_path[dataset_name][0]
    write_path = root_path + data_path[dataset_name][1]
    # TODO MIMIC DATA_AGGREGATOR NEEDS TO BE DONE
    print('model name :', model_name)
    print('dataset name :', dataset_name)
    print('read_path :', read_path)

    if dataset_name == "mimic":
        import vid2bp.preprocessing.MIMICdataset as MIMICdataset
        save_path = write_path + 'case(' + str(degree[-1]) + ')_' + str(chunk_size)
        print('save_path :', save_path)
        ple, abp, dsm = MIMICdataset.data_aggregator(model_name, read_path, chunk_size, samp_rate)
    elif dataset_name == "mimiciii":
        import vid2bp.preprocessing.mimic3temp as mimic3temp
        save_path = write_path + 'case(' + str(degree[-1]) + ')_' + str(chunk_size)
        print('save_path :', save_path)
        ple, abp, dsm = mimic3temp.multi_preprocessing(model_name, read_path)
    else:  # uci
        import vid2bp.preprocessing.UCIdataset as UCIdataset
        save_path = write_path + 'case(' + str(degree[-1]) + ')_' + str(chunk_size)
        print('save_path :', save_path)
        ple, abp, dsm = UCIdataset.data_aggregator(model_name, read_path, chunk_size, samp_rate)

    if model_name == 'BPNet':
        if degree[0] in [1, 2, 3]:
            if degree[0] == 1:
                ple_total = ple
                print('*** P data aggregation done***')
            elif degree[0] == 2:
                ple_first = mm.diff_np(ple)
                ple_total = mm.diff_channels_aggregator(ple, ple_first)
                print('*** P+V data aggregation done***')
            else:
                ple_first = mm.diff_np(ple)
                ple_second = mm.diff_np(ple_first)
                ple_total = mm.diff_channels_aggregator(ple, ple_first, ple_second)
                print('*** P+V+A data aggregation done***')
            tu.data_shuffler(model_name, save_path, [ple_total, abp, dsm], cv)
        else:
            print('not supported derivative ... check MIMIC data_aggregator()')
    elif model_name == 'Unet':
        tu.data_shuffler(model_name, save_path, [ple, abp], cv)
    else:
        print('not supported model... ')

    print('save path :', save_path)


order = orders["second"]
samp_rate = sampling_rate["60"]
selector(model_name="BPNet", dataset_name="uci", degree=order, samp_rate=samp_rate, cv=1)

def total_channel_dataset():
    total = [orders["zeroth"], orders["first"], orders["second"]]
    samp_rate = sampling_rate["60"]
    for t in total:
        selector(model_name="BPNet", dataset_name="uci", degree=t, samp_rate=samp_rate, cv=1)

# total_channel_dataset()
