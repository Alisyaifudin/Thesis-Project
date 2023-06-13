import numpy as np
from os.path import join, abspath
from hammer import dm
from time import time
import pathlib
import vaex
import sys
current = pathlib.Path(__file__).parent.resolve()
root_dir = abspath(join(current, '..', '..', '..'))
sys.path.append(root_dir)
from utils import concat
print(root_dir)
baryon_dir = join(root_dir, 'Data', 'Baryon')
df_baryon = vaex.open(join(baryon_dir, "baryon.hdf5"))
# initializations
# Baryonic density, check table 1 from this https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.121.081101
rhob = df_baryon["rho"].to_numpy()  # Msun/pc^3
sigmaz = df_baryon["sigma_z"].to_numpy()  # km/s
rhoDM = 0.02
log_nu0 = 0
R = 3.4E-3
zsun = 30
theta = concat(rhob, sigmaz, rhoDM, log_nu0, R, zsun)
w0 = -7.
sigma1 = 10.
sigma2 = 15.
log_sigmaw1 = np.log(sigma1)
log_sigmaw2 = np.log(sigma2)

a1 = 1.
a2 = 0.2
log_a1 = np.log(a1)
log_a2 = np.log(a2)
psi = concat(w0, log_sigmaw1, log_sigmaw2, log_a1, log_a2)

# number of walkers
N = 16

z = np.random.randn(N)*200
w = np.random.randn(N)*20
dz = 1.
pos = np.array([z, w]).T

nwalkers = pos.shape[0]
ndim = pos.shape[1]

t0 = time()
chain = dm.sample(1_001_000, nwalkers, pos, theta, psi,
                  dz=dz, verbose=True, parallel=True)
print(time() - t0, "s")
np.save(join(root_dir, 'Data', 'MCMC-no', 'mock',
        'data', 'mock', 'chain.npy'), chain)
