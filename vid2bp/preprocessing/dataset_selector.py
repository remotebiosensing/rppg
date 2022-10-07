import json
import h5py

with open('/home/paperc/PycharmProjects/Pytorch_rppgs/vid2bp/config/parameter.json') as f:
    json_data = json.load(f)
    param = json_data.get("parameters")
    root_path = json_data.get("parameters").get("root_path")  # mimic uci containing folder
    data_path = json_data.get("parameters").get("dataset_path")  # raw mimic uci selection
    orders = json_data.get("parameters").get("in_channels")
    sampling_rate = json_data.get("parameters").get("sampling_rate")
    models = json_data.get("parameters").get("models")


def selector(model_name, dataset_name, degree, samp_rate, cv=0):
    print('< Blood Pressure Estimation dataset selector >')
    read_path = root_path + data_path[dataset_name][0]
    write_path = root_path + data_path[dataset_name][1]
    # TODO MIMIC DATA_AGGREGATOR NEEDS TO BE DONE
    print('model name :', model_name)
    print('dataset name :', dataset_name)
    print('read_path :', read_path)
    print('write_path :', write_path)

    if dataset_name == "mimic":
        import MIMICdataset
        print("mimic dataset selected")

    else:  # uci
        import UCIdataset
        print("uci dataset selected")
        save_path = write_path + 'case(' + str(degree[-1]) + ')_' + str(int(param['chunk_size'] / sampling_rate['base']) * samp_rate)
        UCIdataset.data_aggregator(model_name, read_path, save_path, degree[1], samp_rate, cv)


order = orders["zero"]
samp_rate = sampling_rate["60"]
selector(model_name="Unet", dataset_name="uci_unet", degree=order, samp_rate=samp_rate, cv=1)
