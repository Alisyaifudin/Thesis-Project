from matplotlib import pyplot as plt
import numpy as np
from time import time
from os.path import abspath, join
import sys
from glob import glob 
import vaex
from hammer import dm
root_dir = abspath(join('..', '..'))
root_data_dir = join(root_dir, 'Data')
sys.path.append(root_dir)
from utils import (plot_corner, plot_chain, plot_fit, style, calculate_probs, get_params,
                   get_initial_position_normal, concat, get_data, generate_init)
style()

zpath = join(root_data_dir, 'MCMC-no', 'mock', 'data',  'z')
wpath = join(root_data_dir, 'MCMC-no', 'mock', 'data', 'z')

name = "Baryon"
baryon_dir = join(root_data_dir, name)

# load baryons components
df_baryon = vaex.open(join(baryon_dir, "baryon.hdf5"))

index = 0
zdata, wdata = get_data(zpath, wpath, index)

pred = np.load(join(zpath, 'PHI', 'pred.npy'))
phi_max = 500
phis = np.linspace(1, phi_max, 1000)
kin = (phis, pred)

theta, locs, scales, labels, labs = generate_init("DM")

ndim = len(locs)+24
nwalker = 10*ndim
p0 = dm.generate_p0(nwalker, locs, scales)
indexes = list(range(ndim))
# print([(loc, th, scale+loc) for loc, th, scale in zip(locs, theta, scales)])
print(theta.shape)
u = dm.log_prob(theta, zdata, kin, locs, scales, dz=1.)
print("u", u)