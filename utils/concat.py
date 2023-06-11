import numpy as np

def concat(*args):
    arr = np.array([])
    for arg in args:
        arg = np.atleast_1d(arg)
        arr = np.append(arr, arg)
    return arr