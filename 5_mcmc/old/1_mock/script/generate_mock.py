import numpy as np
from os.path import join, abspath
from hammer import Model
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
sigmaz = df_baryon["sigma_z"].to_numpy() # km/s
R = 3.4E-3
w0 = -7
sigmaw1 = 5
sigmaw2 = 10
log_sigmaw = np.log(sigmaw1)
q_sigmaw = sigmaw1/sigmaw2
a1 = 1
a2 = 0.1
log_a = np.log(a1)
q_a = a2/a1

psi = concat(rhob, sigmaz, R, w0, log_sigmaw, q_sigmaw, log_a, q_a)

rhoDM = 0.02
log_nu0 = 0
zsun = 30

theta = concat(rhoDM, log_nu0, zsun)

# number of walkers
N = 16

z = np.random.randn(N)*200
w = np.random.randn(N)*20
dz = 0.5
pos = np.array([z, w]).T

nwalkers = pos.shape[0]
ndim = pos.shape[1]

t0 = time()
chain = Model.DM.sample(1_001_000, nwalkers, pos, theta, psi, 
                  dz=dz, verbose=True, parallel=True)
print(time() - t0, "s")
np.save(join(root_dir, 'Data', 'MCMC-no', 'mock',
        'data', 'mock', 'chain.npy'), chain[1000:, :, :])