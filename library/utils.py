import numpy as np

def downsample_to_N(array, N, random=False):
    if len(array) <= N:
        return array

    input_was_list = type(array)==list

    array = np.array(array)

    if not random:
            ds = len(array) // N
            array = array[::ds]
    else:
        array = np.random.choice(array, size=N, replace=False)

    if input_was_list:
        array = array.tolist()

    return array