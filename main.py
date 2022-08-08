from preprocessing import dataset

root_path = '/tmp/pycharm_project_811/VBPNet/dataset/mimic-database-1.0.0/'
path = '039/03900001'
data_path = root_path + path
print('data_path:', data_path)

test_record = dataset.read_data(data_path, 0, 100)

print(type(test_record))

df = test_record.p_signal

print(df[:10])
print(len(df))

signal_name = test_record.sig_name
print(signal_name)
