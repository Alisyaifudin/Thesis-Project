import sys
from os.path import join, abspath, pardir
from glob import glob
import numpy as np
import pandas as pd
import vaex
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from time import time
from multiprocessing import Pool
import emcee
import corner
from scipy.stats import norm, uniform

# ================================================
# ================================================
root_dir = abspath(pardir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from utils import safe_mkdir, nu_mod, fw, style

# constants
dpi = 100
# style
style()
# ================================================
# ================================================
# ================================================
# input from user
index = int(sys.argv[1])
step = int(sys.argv[2])
cores = int(sys.argv[3])
simple = False if sys.argv[4] == "False" else True
# ************************************************
# ************************************************
# ************************************************
# dir paths
root_data_dir = abspath(join(root_dir, "Data"))
data_dir = join(root_data_dir, "MCMC-no")
data_baryon_dir = join(root_data_dir, "Baryon")
data_velocity_dir = join(root_data_dir, "Velocity-Distribution-2")
data_eff_dir = join(root_data_dir, "Effective-Volume-2")
# ================================================
# ================================================
# ================================================
# load baryons components
df_baryon = vaex.open(join(data_baryon_dir, "baryon.hdf5"))
rhos = df_baryon["rho"].to_numpy()  # Msun/pc^3
sigmaz = df_baryon["sigma_z"].to_numpy() # km/s
e_rhos = df_baryon["e_rho"].to_numpy()  # Msun/pc^3
e_sigmaz = df_baryon["e_sigma_z"].to_numpy() # km/s
# ************************************************
# ************************************************
# ************************************************
# load data from files
number_files = glob(join(data_eff_dir, "*.hdf5"))
number_files.sort()
velocity_files = glob(join(data_velocity_dir, "gaia_*.hdf5"))
velocity_files.sort()
velocity_popt_files = glob(join(data_velocity_dir, "popt_gaia_*.hdf5"))
velocity_popt_files.sort()
name = number_files[index].split('/')[-1]
asset_dir = join(data_dir, "assets-no-full-"+name[:-5])
safe_mkdir(asset_dir)
print(asset_dir)
# ================================================
# ================================================
# ================================================
# probs function
from utils import log_posterior_no
# ================================================
# utils functions
from utils import (load_data, plot_data, initialize_prior_no,
                   initialize_walkers_no, run_mcmc, consume_samples_no,
                   plot_corner, plot_fitting_no, get_dataframe_no)
# ************************************************
# ************************************************
# ************************************************
# main program
## load data
data, dim, w0, sigma_w, a_raw, name = load_data(index, number_files, velocity_files, velocity_popt_files)
print(name)
## plot data
plot_data(data, path=join(asset_dir, "data-plot.png"))
## initialize prior
locs, scales, uni_list, norm_list = initialize_prior_no(dim, w0, sigma_w, a_raw, simple=simple)
# initialize walkers
p0, ndim, nwalkers = initialize_walkers_no(locs, scales, dim, simple=simple)
sampler = run_mcmc(
    nwalkers, ndim, p0, dim, log_posterior_no, consume_samples_no, 
    args=(data, locs, scales, dim, norm_list, uni_list, simple), 
    cores=cores, plot=True, step=2000, simple=simple, path=join(asset_dir, "chain-plot-0.png")
)
# run again to burn the last run
next_p0 = sampler.get_chain()[-1]
sampler_new = run_mcmc(
    nwalkers, ndim, next_p0, dim, log_posterior_no, consume_samples_no, 
    args=(data, locs, scales, dim, norm_list, uni_list, simple), 
    cores=16, plot=True, step=step, simple=simple, path=join(asset_dir, "chain-plot-1.png")
)
# plot corner
samples = sampler_new.get_chain()
plot_corner(samples, dim, consume_samples_no, simple=simple, path=join(asset_dir, "corner-plot.png"))
# plot fitting
plot_fitting_no(sampler_new, data, dim, simple=simple, n=500, path=join(asset_dir, "fitting-plot.png"))
# convert to dataframe
df = get_dataframe_no(sampler_new, dim, locs, scales, norm_list, uni_list, nwalkers, simple=simple)
print(df.shape)
# save to hdf5
df.export(join(data_dir, f"no_full_{name}"), progress=True)