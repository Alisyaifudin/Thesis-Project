from matplotlib import pyplot as plt
from time import time
import numpy as np
import vaex
from tqdm import tqdm
from .gravity import nu_mod
from .vvd import fw
from os.path import join, abspath, dirname
import pandas as pd
from .probability import log_prior

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

def initialize_prior_dd_dm(dim, w0, sigma_w, a_raw, simple=True):
    uni_list_DM = ['rhoDM', 'sigmaDD', 'log_hDD', 'log_nu0', 'zsun', 'log_sigma_w', 'w0', 'a']
    norm_list_DM = ['R']
    
    locs = dict(
        rhoDM=-0.03,
        sigmaDD=0.1,
        log_hDD=2,
        log_nu0=np.log(1E-6), 
        zsun=-40, 
        R=3.4E-3, 
        w0=w0-20, 
        log_sigma_w=np.log(sigma_w*0.3), 
        a=a_raw*0.5
    )
    
    scales = dict(
        rhoDM=0.1-locs['rhoDM'],
        sigmaDD=50-locs['sigmaDD'],
        log_hDD=np.log(100)-locs['log_hDD'],
        log_nu0=np.log(1E-4)-locs['log_nu0'], 
        zsun=40-locs['zsun'], 
        R=0.6E-3, 
        w0=np.repeat(40, dim), 
        log_sigma_w=np.log(sigma_w*1.7)-locs['log_sigma_w'], 
        a=np.abs(a_raw*1)
    )
    if not simple:
        norm_list_DM += ['rhos', 'sigmaz']
        locs['rhos'] = rhos
        locs['sigmaz'] = sigmaz
        scales['rhos'] = e_rhos
        scales['sigmaz'] = e_sigmaz

    return locs, scales, uni_list_DM, norm_list_DM

# ========================================================================
# ========================================================================
def initialize_walkers_dd_dm(locs, scales, dim, simple=True):
    theta = np.concatenate([np.ravel(x) for x in locs.values()])
    ndim = len(theta)
    nwalkers = ndim*2+1

    rhoDM_0 = np.random.uniform(low=locs['rhoDM'], high=locs['rhoDM']+scales['rhoDM'], size=nwalkers)
    sigmaDD_0 = np.random.uniform(low=locs['sigmaDD'], high=locs['sigmaDD']+scales['sigmaDD'], size=nwalkers)
    log_hDD_0 = np.random.uniform(low=locs['log_hDD'], high=locs['log_hDD']+scales['log_hDD'], size=nwalkers)
    log_nu0_0 = np.random.uniform(low=locs['log_nu0'], high=locs['log_nu0']+scales['log_nu0'], size=nwalkers)
    zsun_0 = np.random.uniform(low=locs['zsun'], high=locs['zsun']+scales['zsun'], size=nwalkers)
    R_0 = np.random.normal(loc=locs['R'], scale=scales['R'], size=nwalkers)
    w0_0 = np.random.uniform(low=locs['w0'], high=locs['w0']+scales['w0'], size=(nwalkers, dim))
    log_sigma_w_0 = np.random.uniform(low=locs['log_sigma_w'], high=locs['log_sigma_w']+scales['log_sigma_w'], size=(nwalkers, dim))
    a_0 = np.random.uniform(low=locs['a'], high=locs['a']+scales['a'], size=(nwalkers, dim))

    p0 = []

    if not simple:
        rhos_0 = np.random.normal(loc=locs['rhos'], scale=scales['rhos'], size=(nwalkers, 12))
        sigmaz_0 = np.random.normal(loc=locs['sigmaz'], scale=scales['sigmaz'], size=(nwalkers, 12))
        p0 = np.array([*rhos_0.T, *sigmaz_0.T, rhoDM_0, sigmaDD_0, log_hDD_0, log_nu0_0, zsun_0, R_0, *w0_0.T, *log_sigma_w_0.T, *a_0.T]).T
    else:
        p0 = np.array([rhoDM_0, sigmaDD_0, log_hDD_0, log_nu0_0, zsun_0, R_0, *w0_0.T, *log_sigma_w_0.T, *a_0.T]).T
    return p0, ndim, nwalkers

# ========================================================================
# ========================================================================
def consume_samples_dd_dm(samples_raw, dim, simple=True, flatten=False):
    skip = 0
    tot = 9
    labels = []
    if flatten:
        labels = [
            r"$\rho_{b}\times 10^2 [M_{\odot}$ pc$^{-3}]$",
            r"$\rho_{d}\times 10^2 [M_{\odot}$ pc$^{-3}]$",
             r"$\rho_{DM}\times 10^2 [M_{\odot}$ pc$^{-3}]$",
            r"$\sigma_{DD}\ [M_{\odot}$ pc$^{-2}]$",
            r"$h_{DD}$ [pc]",
            r"$\nu_0\times 10^5 [$pc$^{-3}]$", 
            r"$z_{\odot}$ [pc]", 
            r"$R\times 10^3 [M_{\odot}$ pc$^{-3}]$",
            r"$w_0$ [km/s]",
            r"$\sigma_{w}$ [km/s]"]
    else:
        labels = [
            r"$\rho_{b}\times 10^2 [M_{\odot}$ pc$^{-3}]$",
            r"$\rho_{d}\times 10^2 [M_{\odot}$ pc$^{-3}]$",
             r"$\rho_{DM}\times 10^2 [M_{\odot}$ pc$^{-3}]$",
            r"$\sigma_{DD}\ [M_{\odot}$ pc$^{-2}]$",
            r"$\log\ h_{DD}$ [pc]",
            r"$\log \nu_0\times [$pc$^{-3}]$", 
            r"$z_{\odot}$ [pc]", 
            r"$R\times 10^3 [M_{\odot}$ pc$^{-3}]$",
            r"$w_0$ [km/s]",
            r"$\log\ \sigma_{w}$ [km/s]"]
    rho_b = 0
    if simple:
        labels = labels[1:]
    else:
        tot += 1
        rho_b = np.sum(samples_raw[:, :, 0:12], axis=2)
        skip = 24 
    sh = samples_raw.shape
    samples = np.concatenate((samples_raw, np.zeros((sh[0], sh[1], 1))), axis=2).copy()
    samples = samples[:, :, skip:skip+tot]
    w0 = samples_raw[:, :, skip+6:skip+6+dim]
    log_sigma_w = samples_raw[:, :, skip+6+dim:skip+6+2*dim]
    a = samples_raw[:, :, skip+6+2*dim:skip+6+3*dim]

    samples[:, :, 0] = samples[:, :, 0]/1E-2 # rhoDM
    samples[:, :, 5] = samples[:, :, 5]/1E-3 # R
    samples[:, :, 6] = np.average(w0, axis=2, weights=a) # w0
    rho_d = samples[:, :, 1]/(4*np.exp(samples[:, :, 2])) #effective rhoDM from DD
    if flatten:
        sigma_w = np.exp(log_sigma_w)
        samples[:, :, 2] = np.exp(samples[:, :, 2]) # hDD
        samples[:, :, 3] = np.exp(samples[:, :, 3])/1E-5 # nu0
        samples[:, :, 7] = np.average(sigma_w, axis=2, weights=a) # sigma_w
    else:
        samples[:, :, 7] = np.average(log_sigma_w, axis=2, weights=a) # sigma_w
    if not simple:
        # shift
        for i in range(tot-1,1, -1):
            samples[:, :, i] = samples[:, :, i-2]
        samples[:, :, 1] = samples[:, :, 2] + rho_d/1E-2
        samples[:, :, 0] = rho_b/1E-2
    else:
        for i in range(tot-1,0, -1):
            samples[:, :, i] = samples[:, :, i-1]
        samples[:, :, 0] = samples[:, :, 1] + rho_d/1E-2
    if flatten:
        samples = samples.reshape(-1, tot)
    return samples, labels

# ========================================================================
# ========================================================================

def plot_fitting_dd_dm(sampler, data, dim, alpha=0.01, n=200, log=False, path=None, simple=True, dpi=150):
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
    axes[0].set_ylabel(r'$\nu(z)$')
    axes[0].legend()
    axes[1].errorbar(wmid, wdens, yerr=werr, fmt='.', label='data')
    axes[1].set_xlabel(r'w')
    axes[1].set_ylabel(r'fw')
    axes[1].legend()
    skip = 0 if simple else 24
    if log:
        axes[0].set_yscale('log')
        axes[1].set_yscale('log')
    for i in tqdm(range(n)):
        index = np.random.randint(0, len(flat_samples))
        theta_dict = dict(
            rhoDM=flat_samples[index, skip],
            sigmaDD=flat_samples[index, skip+1],
            hDD=np.exp(flat_samples[index, skip+2]),
            nu0=np.exp(flat_samples[index, skip+3]),
            zsun=flat_samples[index, skip+4],
            R=flat_samples[index, skip+5],
            w0=flat_samples[index, skip+6:skip+6+dim],
            sigma_w=np.exp(flat_samples[index, skip+6+dim:skip+6+2*dim]),
            a=flat_samples[index, skip+6+2*dim:skip+6+3*dim]
        )
        if not simple:
            theta_dict['rhos'] = flat_samples[index, :12]
            theta_dict['sigmaz'] = flat_samples[index, 12:24]
        else:
            theta_dict['rhos'] = rhos
            theta_dict['sigmaz'] = sigmaz
        
        nu = nu_mod(zs, **theta_dict)
        axes[0].plot(zs, nu, label='model', c="r", alpha=alpha)
        Fw = fw(ws, **theta_dict)
        axes[1].plot(ws, Fw, label='model', c="r", alpha=alpha)
    if path is not None:
        fig.savefig(path, dpi=dpi)
    fig.show()

# ========================================================================
# ========================================================================

def get_dataframe_dd_dm(sampler_new, dim, locs, scales, norm_list_DM, uni_list_DM, nwalkers, simple=True):
    flat_samples = sampler_new.get_chain().copy()
    log_posterior_chain = sampler_new.get_log_prob().copy()
    thetas = []
    t0 = time()
    skip = 0 if simple else 24
    for i, post in enumerate(log_posterior_chain.T):
        for samples, lg_posterior in zip(tqdm(flat_samples[:, i], desc=f"{i}/{nwalkers}", leave=False), post):
            theta = {
                "rhoDM": samples[skip],
                "sigmaDD": samples[skip+1],
                "log_hDD": samples[skip+2],
                "log_nu0": samples[skip+3],
                "zsun": samples[skip+4],
                "R": samples[skip+5],
                "w0": samples[skip+6:skip+6+dim],
                "log_sigma_w": samples[skip+6+dim:skip+6+2*dim],
                "a": samples[skip+6+2*dim:skip+6+3*dim]
            }
            if simple:
                theta['rhos'] = rhos
                theta['sigmaz'] = sigmaz
            else:
                theta['rhos'] = samples[:12]
                theta['sigmaz'] = samples[12:24]
            lg_prior = log_prior(theta, locs, scales, norm_list_DM, uni_list_DM)
            theta['log_prior'] = lg_prior
            theta['log_posterior'] = lg_posterior
            theta['log_likelihood'] = lg_posterior - lg_prior
            del theta['w0'], theta['log_sigma_w'], theta['a'], theta['rhos'], theta['sigmaz']
            for j in range(dim):
                theta[f"w0_{j}"] = samples[skip+5+j]
                theta[f"log_sigma_w_{j}"] = samples[skip+5+dim+j]
                theta[f"a_{j}"] = samples[skip+5+2*dim+j]
            if not simple:
                for j in range(12):
                    theta[f"rhob_{j}"] = samples[j]
                    theta[f"sigmaz_{j}"] = samples[12+j]
            theta['walker'] = i
            thetas.append(theta)
    t1 = time()
    print(f"Time: {t1-t0:.2f} s")
    df = pd.DataFrame(thetas)
    ## save
    df_dm = vaex.from_pandas(df)
    return df_dm