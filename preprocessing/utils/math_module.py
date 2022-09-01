import numpy as np

'''should be done right after sig_slicing()'''

'''
np.shape(input_sig) : ndarray(702, 7500)     ex) ple, abp
'''

# TODO np.diff 사용 고려


def diff_np(input_sig, input_sig2=None):
    ple_diff = []
    abp_diff = []
    if input_sig2 is None:
        for p in input_sig:
            ple_temp = np.append(p[1:], p[-1]) - p
            ple_temp[-1] = np.mean(ple_temp[-3:-2])
            ple_diff.append(ple_temp)

        ple_diff = np.array(ple_diff)
        return ple_diff
    else:
        for p, s in zip(input_sig, input_sig2):
            ple_temp = np.append(p[1:], p[-1]) - p
            abp_temp = np.append(s[1:], s[-1]) - s
            ple_temp[-1] = np.mean(ple_temp[-3:-2])
            abp_temp[-1] = np.mean(abp_temp[-3:-2])
            ple_diff.append(ple_temp)
            abp_diff.append(abp_temp)

        ple_diff = np.array(ple_diff)
        abp_diff = np.array(abp_diff)
        return ple_diff, abp_diff


# TODO -> data_aggregator에서 degree에 따른 결과가 가지 수가 안맞음
def diff_channels_aggregator(zero, first=None, second=None):
    zero = np.expand_dims(zero, axis=1)

    if first is None:
        # print('zero called')
        print('channel aggregated ( f ) :', np.shape(zero))
        # print(zero[0])
        return zero

    elif (first is not None) and (second is None):
        # print('first called')
        first = np.expand_dims(first, axis=1)
        temp1 = np.concatenate((zero, first), axis=1)

        print('channel aggregated ( f + f\' ) :', np.shape(temp1))
        # print(temp1[0])
        return temp1
    elif (first is not None) and (second is not None):
        # print('second called')
        first = np.expand_dims(first, axis=1)
        second = np.expand_dims(second, axis=1)
        temp2 = np.concatenate((zero, first, second), axis=1)

        print('channel aggregated ( f + f\' + f\'\' ) :', np.shape(temp2))
        # print(temp2[0])
        # print(temp2)
        return temp2
