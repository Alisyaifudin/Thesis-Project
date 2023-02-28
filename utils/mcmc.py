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
        fig.savefig(path, dpi=300)

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
        plt.savefig(path, dpi=300)
    plt.show()

def initialize_prior_dm(w0, sigma_w, a_raw, dim):
    locs = dict(
        log_rhoDM=np.log(0.001), 
        log_nu0=np.log(1E-5), 
        zsun=-20, 
        R=3.4E-3, 
        w0=w0-5, 
        log_sigma_w=np.log(sigma_w*0.7), 
        a=np.select([a_raw > 0], [a_raw*0.7], default=a_raw*1.3)
    )

    scales = dict(
        log_rhoDM=np.log(0.1)-locs['log_rhoDM'], 
        log_nu0=np.log(1E-4)-locs['log_nu0'], 
        zsun=20-locs['zsun'], 
        R=0.6E-3, 
        w0=np.repeat(10, dim), 
        log_sigma_w=np.log(sigma_w*1.3)-locs['log_sigma_w'], 
        a=np.abs(a_raw*0.6)
    )
    uni_list_DM = ['log_rhoDM', 'log_nu0', 'zsun', 'log_sigma_w', 'w0', 'a']
    norm_list_DM = ['R']
    return locs, scales, uni_list_DM, norm_list_DM

def initialize_walkers_dm(locs, scales, dim):
    theta = np.concatenate([np.ravel(x) for x in locs.values()])
    ndim = len(theta)
    nwalkers = ndim*2+1

    log_rhoDM_0 = np.random.uniform(low=locs['log_rhoDM'], high=locs['log_rhoDM']+scales['log_rhoDM'], size=nwalkers)
    log_nu0_0 = np.random.uniform(low=locs['log_nu0'], high=locs['log_nu0']+scales['log_nu0'], size=nwalkers)
    zsun_0 = np.random.uniform(low=locs['zsun'], high=locs['zsun']+scales['zsun'], size=nwalkers)
    R_0 = np.random.normal(loc=locs['R'], scale=scales['R'], size=nwalkers)
    w0_0 = np.random.uniform(low=locs['w0'], high=locs['w0']+scales['w0'], size=(nwalkers, dim))
    log_sigma_w_0 = np.random.uniform(low=locs['log_sigma_w'], high=locs['log_sigma_w']+scales['log_sigma_w'], size=(nwalkers, dim))
    a_0 = np.random.uniform(low=locs['a'], high=locs['a']+scales['a'], size=(nwalkers, dim))

    p0 = np.array([log_rhoDM_0, log_nu0_0, zsun_0, R_0, *w0_0.T, *log_sigma_w_0.T, *a_0.T]).T
    return p0, ndim, nwalkers

def run_mcmc(nwalkers, ndim, p0, labels, log_posterior, args, cores=16, plot=False, step=500, start=0, skip=0, path=None):
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
        start_t = time()
        sampler.run_mcmc(p0, step, progress=True)
        samples = sampler.get_chain()
        end = time()
        multi_time = end - start_t
        if plot:
            plot_chain(samples, labels, figsize=(10,15), start=start, skip=skip, path=path)
        print("Multiprocessing took {0:.1f} seconds".format(multi_time))
        sampler_ = sampler
    return sampler_

def plot_corner_dm(sampler, path=None):
    flat_samples = sampler.get_chain(flat=True).copy()
    flat_samples = flat_samples[:, :4]
    flat_samples[:, 0] = np.exp(flat_samples[:, 0])/1E-2
    flat_samples[:, 1] = np.exp(flat_samples[:, 1])/1E-5
    flat_samples[:, 3] = flat_samples[:, 3]/1E-3

    labels = [r"$\rho_{DM}\times 10^2$",r"$\nu_0 10^5$ " , r"$z_{\odot}$", r"$R\times 10^3$"]
    fig = corner.corner(
        flat_samples, labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True, title_kwargs={"fontsize": 12},
    )
    if path is not None:
        fig.savefig(path, dpi=300)

def plot_fitting_dm(sampler, data, dim, rhos, sigmaz, alpha=0.01, n=200, path=None):
    zdata, wdata = data
    zmid, zdens, zerr = zdata
    wmid, wdens, werr = wdata
    # plot two subplot
    flat_samples = sampler.get_chain(flat=True).copy()
    zs = np.linspace(-200, 200, 100)
    ws = np.linspace(-70, 70, 1000)

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    axes[0].errorbar(zmid, zdens, yerr=zerr, fmt='o', label='data')
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
            rhoDM=np.exp(flat_samples[index, 0]),
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
        fig.savefig(path, dpi=300)
    plt.show()