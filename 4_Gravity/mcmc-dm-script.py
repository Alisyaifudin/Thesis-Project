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
root_dir = abspath(pardir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from utils import safe_mkdir, nu_mod, fw, style

# style
style()
# constants
dpi = 100
# ================================================
# ================================================
# ================================================
# input from user
index = int(sys.argv[1])
step = int(sys.argv[2])
cores = int(sys.argv[3])
# ************************************************
# ************************************************
# ************************************************
# dir paths
root_data_dir = abspath(join(root_dir, "Data"))
data_dir = join(root_data_dir, "MCMC-2")
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
asset_dir = join(data_dir, "assets-dm-"+name[:-5])
safe_mkdir(asset_dir)
print(asset_dir)
# ================================================
# ================================================
# ================================================
# probs function
## prior
def log_prior(theta, locs, scales, norm_list, uni_list):
    pars_list = norm_list+uni_list
    result = 0
    for item in pars_list:
        if item in uni_list:
            result += np.sum(uniform.logpdf(theta[item], loc=locs[item], scale=scales[item]))
        elif item in norm_list:
            result += np.sum(norm.logpdf(theta[item], loc=locs[item], scale=scales[item]))
    return result
## likelihood
def log_likelihood(theta, zdata, wdata):
    zmid, zdens, zerr = zdata
    wmid, wdens, werr = wdata
    Fz = nu_mod(zmid, **theta)
    Fw = fw(wmid, **theta)
    resz = np.sum(norm.logpdf(zdens, loc=Fz, scale=zerr))
    resw = np.sum(norm.logpdf(wdens, loc=Fw, scale=werr))
    return resz + resw
## posterior DM only
def log_posterior_simple_DM(theta, data, locs, scales, dim, norm_list, uni_list):
    zdata, wdata = data
    theta_dict = dict(
        rhos=rhos,
        sigmaz=sigmaz,
        rhoDM=theta[0],
        log_nu0=theta[1],
        zsun=theta[2],
        R=theta[3],
        w0=theta[4:4+dim],
        log_sigma_w=theta[4+dim:4+2*dim],
        a=theta[4+2*dim:4+3*dim]
    )
    log_prior_ = log_prior(theta_dict, locs, scales, norm_list, uni_list)
    if not np.isfinite(log_prior_):
        return -np.inf

    theta_dict['sigmaDD'] = 0
    theta_dict['hDD'] = 1
    theta_dict['nu0'] = np.exp(theta_dict['log_nu0'])
    theta_dict['sigma_w'] = np.exp(theta_dict['log_sigma_w'])
    
    return log_prior_ + log_likelihood(theta_dict, zdata, wdata)
# ================================================
# utils functions
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
# ================================================
def plot_data(data, path=None):
    zdata, wdata = data
    zmid, zdens, zerr = zdata
    wmid, wdens, werr = wdata

    # plot 2 subplot
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    axes[0].errorbar(zmid, zdens, yerr=zerr, fmt='.')
    axes[0].set_xlabel(r'$z$ [pc]')
    axes[0].set_ylabel(r'$\rho(z)$ [pc$^{-3}$]')
    axes[1].errorbar(wmid, wdens, yerr=werr, fmt='.')
    axes[1].set_xlabel(r'$w$ [km/s]')
    axes[1].set_ylabel(r'$f_w$')
    if path is not None:
        fig.savefig(path, dpi=dpi)
    fig.show()
# ================================================
def initialize_prior_simple_dm(w0, sigma_w, a_raw, dim):
    locs = dict(
        rhoDM=-0.05, 
        log_nu0=np.log(1E-6), 
        zsun=-40, 
        R=3.4E-3, 
        w0=w0-10, 
        log_sigma_w=np.log(sigma_w*0.7), 
        a=a_raw*0.7
    )

    scales = dict(
        rhoDM=0.1-locs['rhoDM'], 
        log_nu0=np.log(1E-4)-locs['log_nu0'], 
        zsun=40-locs['zsun'], 
        R=0.6E-3, 
        w0=np.repeat(20, dim), 
        log_sigma_w=np.log(sigma_w*1.3)-locs['log_sigma_w'], 
        a=np.abs(a_raw*0.6)
    )
    uni_list_DM = ['rhoDM', 'log_nu0', 'zsun', 'log_sigma_w', 'w0', 'a']
    norm_list_DM = ['R']
    return locs, scales, uni_list_DM, norm_list_DM
# ================================================
def initialize_walkers_simple_dm(locs, scales, dim):
    theta = np.concatenate([np.ravel(x) for x in locs.values()])
    ndim = len(theta)
    nwalkers = ndim*2+1

    rhoDM_0 = np.random.uniform(low=locs['rhoDM'], high=locs['rhoDM']+scales['rhoDM'], size=nwalkers)
    log_nu0_0 = np.random.uniform(low=locs['log_nu0'], high=locs['log_nu0']+scales['log_nu0'], size=nwalkers)
    zsun_0 = np.random.uniform(low=locs['zsun'], high=locs['zsun']+scales['zsun'], size=nwalkers)
    R_0 = np.random.normal(loc=locs['R'], scale=scales['R'], size=nwalkers)
    w0_0 = np.random.uniform(low=locs['w0'], high=locs['w0']+scales['w0'], size=(nwalkers, dim))
    log_sigma_w_0 = np.random.uniform(low=locs['log_sigma_w'], high=locs['log_sigma_w']+scales['log_sigma_w'], size=(nwalkers, dim))
    a_0 = np.random.uniform(low=locs['a'], high=locs['a']+scales['a'], size=(nwalkers, dim))

    p0 = np.array([rhoDM_0, log_nu0_0, zsun_0, R_0, *w0_0.T, *log_sigma_w_0.T, *a_0.T]).T
    return p0, ndim, nwalkers
# ================================================
def plot_chain(samples, labels, figsize=(10, 7), alpha=0.3, start=0, skip=0, path=None):
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
# ================================================
def run_mcmc(nwalkers, ndim, p0, labels, log_posterior, args, cores=16, plot=False, step=500, skip=0, start=0, path=None):
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
        samples = sampler.get_chain()
        t1 = time()
        multi_time = t1 - t0
        if plot:
            plot_chain(samples, labels, figsize=(10,15), skip=skip, start=start, path=path)
        print("Multiprocessing took {0:.1f} seconds".format(multi_time))
        sampler_ = sampler
    return sampler_
# ================================================
def plot_corner_dm(sampler, path=None):
    flat_samples = sampler.get_chain(flat=True).copy()
    flat_samples = flat_samples[:, :4]
    flat_samples[:, 0] = flat_samples[:, 0]/1E-2
    flat_samples[:, 1] = np.exp(flat_samples[:, 1])/1E-5
    flat_samples[:, 3] = flat_samples[:, 3]/1E-3

    labels = [r"$\rho_{DM}\times 10^2 [M_{\odot}\textup{ pc}^{-3}]$",r"$\nu_0 \times 10^5 [\textup{pc}^{-3}]$ " , r"$z_{\odot}$ [\textup{ pc}]", r"$R\times 10^3 [M_{\odot}\textup{ pc}^{-3}]$"]
    fig = corner.corner(
        flat_samples, labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True, title_kwargs={"fontsize": 12},
    )
    if path is not None:
        fig.savefig(path, dpi=dpi)
    fig.show()
# ================================================
def plot_fitting_dm(sampler, data, dim, rhos, sigmaz, alpha=0.01, n=200, path=None):
    zdata, wdata = data
    zmid, zdens, zerr = zdata
    wmid, wdens, werr = wdata
    # plot two subplot
    flat_samples = sampler.get_chain(flat=True).copy()
    zs = np.linspace(zmid.min(), zmid.max(), 100)
    ws = np.linspace(wmid.min(), wmid.max(), 100)

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    axes[0].errorbar(zmid, zdens, yerr=zerr, fmt='.', label='data')
    axes[0].set_xlabel(r'$z$')
    axes[0].set_ylabel(r'$\rho(z)$')
    axes[0].legend()
    axes[1].errorbar(wmid, wdens, yerr=werr, fmt='.', label='data')
    axes[1].set_xlabel(r'w')
    axes[1].set_ylabel(r'num')
    axes[1].legend()
    for i in tqdm(range(n)):
        index = np.random.randint(0, len(flat_samples))
        theta_dict = dict(
            rhos=rhos, 
            sigmaz=sigmaz, 
            rhoDM=flat_samples[index, 0],
            sigmaDD=0, 
            hDD=1, 
            nu0=np.exp(flat_samples[index, 1]),
            zsun=flat_samples[index, 2],
            R=flat_samples[index, 3],
            w0=flat_samples[index, 4:4+dim],
            sigma_w=np.exp(flat_samples[index, 4+dim:4+2*dim]),
            a=flat_samples[index, 4+2*dim:4+3*dim]
        )
        nu = nu_mod(zs, **theta_dict)
        axes[0].plot(zs, nu, label='model', c="r", alpha=alpha)
        Fw = fw(ws, **theta_dict)
        axes[1].plot(ws, Fw, label='model', c="r", alpha=alpha)
    if path is not None:
        fig.savefig(path, dpi=dpi)
    fig.show()
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
locs, scales, uni_list_DM, norm_list_DM = initialize_prior_simple_dm(w0, sigma_w, a_raw, dim)
## initialize walkers
p0, ndim, nwalkers = initialize_walkers_simple_dm(locs, scales, dim)
## run mcmc
labels = [r"$\rho_{DM} [M_{\odot}\textup{ pc}^{-3}]$",r"$\log \nu_0 [\textup{pc}^{-3}]$ " , r"$z_{\odot}$ [\textup{ pc}]", r"$R [M_{\odot}\textup{ pc}^{-3}]$"]
sampler = run_mcmc(
    nwalkers, ndim, p0, labels, log_posterior_simple_DM,cores=cores,
    args=[data, locs, scales, dim, norm_list_DM, uni_list_DM], 
    plot=True, step=500, path=join(asset_dir, "chain-plot-0.png")
)
## run again, with the last sample as the initial point
next_p0 = sampler.get_chain()[-1]
sampler_new = run_mcmc(
    nwalkers, ndim, next_p0, labels, log_posterior_simple_DM, cores=cores,
    args=[data, locs, scales, dim, norm_list_DM, uni_list_DM], 
    plot=True, step=step, path=join(asset_dir, "chain-plot-1.png")
)
## plot corner
plot_corner_dm(sampler_new, path=join(asset_dir, "corner-plot.png"))
## plot fitting
plot_fitting_dm(sampler_new, data, dim, rhos, sigmaz, alpha=0.05, n=100, path=join(asset_dir, "fitting-plot.png"))
## save sampler into dataframe
flat_samples = sampler_new.get_chain().copy()
log_posterior_chain = sampler_new.get_log_prob().copy()
thetas = []
t0 = time()
for i, post in enumerate(log_posterior_chain.T):
    for samples, lg_posterior in zip(tqdm(flat_samples[:, i], desc=f"{i}/{nwalkers}", leave=False), post):
        theta = {
            "rhoDM": samples[0],
            "log_nu0": samples[1],
            "zsun": samples[2],
            "R": samples[3],
            "w0": samples[4:4+dim],
            "log_sigma_w": samples[4+dim:4+2*dim],
            "a": samples[4+2*dim:4+3*dim]
        }
        lg_prior = log_prior(theta, locs, scales, norm_list_DM, uni_list_DM)
        theta['log_prior'] = lg_prior
        theta['log_posterior'] = lg_posterior
        theta['log_likelihood'] = lg_posterior - lg_prior
        del theta['w0'], theta['log_sigma_w'], theta['a']
        for j in range(dim):
            theta[f"w0_{j}"] = samples[4+j]
            theta[f"log_sigma_w_{j}"] = samples[4+dim+j]
            theta[f"a_{j}"] = samples[4+2*dim+j]
        theta['walker'] = i
        thetas.append(theta)
# for samples in tqdm(flat_samples):
t1 = time()
print(f"Time: {t1-t0:.2f} s")
df = pd.DataFrame(thetas)
## save
df_dm = vaex.from_pandas(df)
df_dm.export(join(data_dir, f"dm_simple_{name}"), progress=True)