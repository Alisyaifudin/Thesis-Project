import numpy as np


# interpolation function
def simple(x):
    return x/(1+x)

def standard(mu):
    return mu/np.sqrt(1+mu**2)


# invers interpolation function
def inv_simple(mu):
    return mu/(1-mu)

def inv_standard(mu):
    return mu/np.sqrt(1-mu**2)