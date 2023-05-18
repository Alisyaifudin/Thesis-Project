from matplotlib import pyplot as plt
import numpy as np
from os.path import join, abspath
from hammer import dm
from time import time
import pathlib

current = pathlib.Path(__file__).parent.resolve()
root_dir = abspath(join(current, '..', '..', '..'))
print(root_dir)

# initializations
rhob = [
    0.0104, 0.0277, 0.0073, 0.0005, 0.0006, 0.0018,
    0.0018, 0.0029, 0.0072, 0.0216, 0.0056, 0.0015
]
sigmaz = [
    3.7, 7.1, 22.1, 39.0, 15.5, 7.5, 12.0,
    18.0, 18.5, 18.5, 20.0, 20.0]
rhoDM = [0.016]
log_nu0 = [1]
R = [3.4E-3]
zsun = [30]
w0 = [-7.]
log_sigmaw1 = [np.log(5.)]
log_a1 = [np.log(.6)]
log_sigmaw2 = [np.log(6.)]
log_a2 = [np.log(0.4)]

theta = np.array([rhob + sigmaz + rhoDM + log_nu0 + R + zsun +
                 w0 + log_sigmaw1 + log_a1 + log_sigmaw2 + log_a2]).flatten()

# number of walkers
N = 16

z = np.random.randn(N)*200
w = np.random.randn(N)*20
dz = 1
pos = np.array([z, w]).T

nwalkers = pos.shape[0]
ndim = pos.shape[1]

t0 = time()
chain = dm.sample(1_000_000, nwalkers, pos, theta,
                  dz=1., verbose=True, parallel=True)
print(time() - t0, "s")
np.save(join(root_dir, 'Data', 'MCMC', 'dm_mock', 'data', 'chain.npy'), chain)
