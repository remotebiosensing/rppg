import numpy as np

test = np.array([5, 4, 3, 2, 1, 2, 3, 4, 5])


def derivative(input_sig):
    # deriv = np.array(len(input_sig))
    temp = input_sig[1:]
    temp = np.append(temp, 0)
    deriv = np.subtract(temp, input_sig)
    print(temp)
    print('deriv :', deriv)
    return deriv


deri = derivative(test)
print(deri)
