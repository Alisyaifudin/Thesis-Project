from matplotlib import pyplot as plt
import numpy as np
from time import time
from os.path import abspath, join
import sys
from glob import glob
import vaex
from tqdm import tqdm
from hammer import vel
from scipy.stats import median_abs_deviation as mad_func
root_dir = abspath(join('..', '..'))
root_data_dir = join(root_dir, 'Data')
sys.path.append(root_dir)
from utils import concat, get_data, generate_init

root_data_dir = abspath(join(root_dir, "Data"))
baryon_dir = join(root_data_dir, "Baryon")
df_baryon = vaex.open(join(baryon_dir, "baryon.hdf5"))

# init
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

zpath = join(root_data_dir, 'MCMC-no', 'mock', 'data', 'z')
wpath = join(root_data_dir, 'MCMC-no', 'mock', 'data', 'z')
ind = 0
zdata, wdata = get_data(zpath, wpath, ind)
files = glob(join(zpath, "z*"))
files.sort()
name_pred = files[ind].split(
    "/")[-1].split(".")[0].replace("z", "pred") + ".npy"
name_phi = files[ind].split("/")[-1].split(".")[0].replace("z", "phi") + ".npy"

psi, locs, scales, labels, labs = generate_init("kin")
ndim = len(labs)
nwalker = 10*ndim
p0 = vel.generate_p0(nwalker, locs, scales)
indexes = list(range(ndim))

chain = vel.mcmc(50, p0, wdata, locs, scales, parallel=True, verbose=True)
print(chain)
