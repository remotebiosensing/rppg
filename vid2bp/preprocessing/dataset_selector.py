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
    chunk_size = json_data.get("parameters").get("chunk_size")


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
        save_path = write_path + 'case(' + str(degree[-1]) + ')_' + str(chunk_size)
        print("mimic dataset selected")

    else:  # uci
        import UCIdataset
        save_path = write_path + 'case(' + str(degree[-1]) + ')_' + str(chunk_size)
        UCIdataset.data_aggregator(model_name, read_path, save_path, degree[1], chunk_size, samp_rate, cv)


order = orders["sixth"]
samp_rate = sampling_rate["60"]
selector(model_name="BPNet", dataset_name="uci", degree=order, samp_rate=samp_rate, cv=1)
