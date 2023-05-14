from matplotlib import pyplot as plt
from multiprocessing import Pool
import emcee 
from time import time
import numpy as np
import vaex
import corner
import os
from tqdm import tqdm
from .gravity import nu_mod
from .vvd import fw
from os.path import join, abspath, dirname
import pandas as pd

current_dir = dirname(abspath(__file__))
root_dir = abspath(join(current_dir, ".."))
root_data_dir = abspath(join(root_dir, "Data"))
data_baryon_dir = join(root_data_dir, "Baryon")
df_baryon = vaex.open(join(data_baryon_dir, "baryon.hdf5"))
rhos = df_baryon["rho"].to_numpy()  # Msun/pc^3
sigmaz = df_baryon["sigma_z"].to_numpy() # km/s
e_rhos = df_baryon["e_rho"].to_numpy()  # Msun/pc^3
e_sigmaz = df_baryon["e_sigma_z"].to_numpy() # km/s

# ========================================================================
# ========================================================================
def load_data(index, number_files, velocity_files, velocity_popt_files):
    name = number_files[index].split('/')[-1]
    df_number = vaex.open(number_files[index])
    df_velocity = vaex.open(velocity_files[index])
    df_popt = vaex.open(velocity_popt_files[index])
    popt = df_popt['popt'].to_numpy()
    wdens= df_velocity['wnum'].to_numpy()
    werr = df_velocity['werr'].to_numpy()
    wmid = df_velocity['w'].to_numpy()
    dim = len(popt)//3

    zdens = df_number['density_corr'].to_numpy()
    zerr = df_number['density_err'].to_numpy()
    zmid = df_number['z'].to_numpy()

    w0 = []
    sigma_w = []
    a_raw = []

    for i in range(len(popt)//3):
        w0_i = popt[3*i+1]
        sigma_w_i = popt[3*i+2]
        a_raw_i = popt[3*i]
        
        w0.append(w0_i)
        sigma_w.append(sigma_w_i)
        a_raw.append(a_raw_i)

    w0 = np.array(w0)
    sigma_w = np.array(sigma_w)
    a_raw = np.array(a_raw)

    zdata = (zmid, zdens, zerr)
    wdata = (wmid, wdens, werr)
    data = (zdata, wdata)
    return data, dim, w0, sigma_w, a_raw, name

# ========================================================================
# ========================================================================

def plot_data(data, path=None, dpi=100):
    zdata, wdata = data
    zmid, zdens, zerr = zdata
    wmid, wdens, werr = wdata

    # plot 2 subplot
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    axes[0].errorbar(zmid, zdens, yerr=zerr, fmt='.', capsize=2, lw=1)
    axes[0].set_xlabel(r'$z$ [pc]')
    axes[0].set_ylabel(r'$\rho(z)$ [pc$^{-3}$]')
    axes[1].errorbar(wmid, wdens, yerr=werr, fmt='.', capsize=2, lw=1)
    axes[1].set_xlabel(r'$w$ [km/s]')
    axes[1].set_ylabel(r'$f_w$')
    if path is not None:
        fig.savefig(path, dpi=dpi)
    fig.show()

# ========================================================================
# ========================================================================

def plot_chain(samples, labels, figsize=(10, 7), alpha=0.3, start=0, skip=0, path=None, dpi=150):
    num = len(labels)
    fig, axes = plt.subplots(num, figsize=figsize, sharex=True)
    if len(labels) == 1:
      axes = [axes]
    for i in range(num):
        ax = axes[i]
        ax.plot(samples[start:, :, skip+i], "k", alpha=alpha)
        ax.set_xlim(0, len(samples[start:, :, skip+i]))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")
    if path is not None:
        fig.savefig(path, dpi=dpi)
    fig.show()


# ========================================================================
# ========================================================================

def run_mcmc(nwalkers, ndim, p0, dim, log_posterior, consume_samples, args, cores=16, plot=False, step=500, start=0, path=None, dpi=150, simple=True):
    os.environ["OMP_NUM_THREADS"] = str(cores)
    sampler_ = 0
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, 
            ndim, 
            log_posterior, 
            args=args, 
            pool=pool
        )
        t0 = time()
        sampler.run_mcmc(p0, step, progress=True)
        samples = sampler.get_chain().copy()
        samples, labels = consume_samples(samples, dim, simple=simple)
        
        t1 = time()
        multi_time = t1 - t0
        if plot:
            plot_chain(samples, labels, figsize=(10,15), start=start, path=path, dpi=dpi)
        print("Multiprocessing took {0:.1f} seconds".format(multi_time))
        sampler_ = sampler
    return sampler_

# ========================================================================
# ========================================================================

def plot_corner(samples, dim, consume_samples, path=None, dpi=150, simple=True):
    flat_samples, labels = consume_samples(samples, dim, flatten=True, simple=simple)

    fig = corner.corner(
        flat_samples, labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True, title_kwargs={"fontsize": 12},
    )
    if path is not None:
        fig.savefig(path, dpi=dpi)
    fig.show()