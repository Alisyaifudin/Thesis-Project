from matplotlib import pyplot as plt
from time import time
import numpy as np
import vaex
from tqdm import tqdm
from .gravity import nu_mod
from .vvd import fw
from .progressbar import progressbar
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

def initialize_prior_no(dim, w0, sigma_w, a_raw, simple=True):
    uni_list = ['log_nu0', 'zsun', 'log_sigma_w', 'w0', 'a']
    norm_list = ['R']
    
    locs = dict(
        log_nu0=np.log(1E-6), 
        zsun=-40, 
        R=3.4E-3, 
        w0=w0-20, 
        log_sigma_w=np.log(sigma_w*0.7), 
        a=a_raw*0.7
    )
    
    scales = dict(
        log_nu0=np.log(1E-4)-locs['log_nu0'], 
        zsun=40-locs['zsun'], 
        R=0.6E-3, 
        w0=np.repeat(40, dim), 
        log_sigma_w=np.log(sigma_w*1.3)-locs['log_sigma_w'], 
        a=np.abs(a_raw*0.6)
    )
    if not simple:
        norm_list += ['rhos', 'sigmaz']
        locs['rhos'] = rhos
        locs['sigmaz'] = sigmaz
        scales['rhos'] = e_rhos
        scales['sigmaz'] = e_sigmaz

    return locs, scales, uni_list, norm_list

# ========================================================================
# ========================================================================
def initialize_walkers_no(locs, scales, dim, simple=True):
    theta = np.concatenate([np.ravel(x) for x in locs.values()])
    ndim = len(theta)
    nwalkers = ndim*2+1

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
        p0 = np.array([*rhos_0.T, *sigmaz_0.T, log_nu0_0, zsun_0, R_0, *w0_0.T, *log_sigma_w_0.T, *a_0.T]).T
    else:
        p0 = np.array([log_nu0_0, zsun_0, R_0, *w0_0.T, *log_sigma_w_0.T, *a_0.T]).T
    return p0, ndim, nwalkers

# ========================================================================
# ========================================================================
def sample_rhob(rhos, e_rhos, nsteps, nwalkers):
    rhob = np.zeros((nsteps, nwalkers, 12))
    for i in range(12):
        rhob[:, :, i] = np.random.normal(loc=rhos[i], scale=e_rhos[i], size=(nsteps, nwalkers))
    return rhob

# ========================================================================
# ========================================================================
def consume_samples_no(samples_raw, dim, simple=True, flatten=False):
    skip = 0
    tot = 5
    labels = []
    if flatten:
        labels = [
            r"$\rho_{b}\times 10^2 [M_{\odot}$ pc$^{-3}]$",
            r"$\rho_{d}\times 10^2 [M_{\odot}$ pc$^{-3}]$",
            r"$\nu_0\times 10^5 [$pc$^{-3}]$", 
            r"$z_{\odot}$ [pc]", 
            r"$R\times 10^3 [M_{\odot}$ pc$^{-3}]$",
            r"$w_0$ [km/s]",
            r"$\sigma_{w}$ [km/s]"]
    else:
        labels = [
            r"$\rho_{b}\times 10^2 [M_{\odot}$ pc$^{-3}]$",
            r"$\rho_{d}\times 10^2 [M_{\odot}$ pc$^{-3}]$",
            r"$\log \nu_0\times [$pc$^{-3}]$", 
            r"$z_{\odot}$ [pc]", 
            r"$R\times 10^3 [M_{\odot}$ pc$^{-3}]$",
            r"$w_0$ [km/s]",
            r"$\log\ \sigma_{w}$ [km/s]"]
    rho_b = 0
    rho_d = 0
    sh = samples_raw.shape
    if simple:
        labels = labels[2:]
    else:
        tot += 2
        rho_b = sample_rhob(rhos, e_rhos, sh[0], sh[1])
        rho_d = samples_raw[:, :, 0:12] - rho_b
        rho_b = np.sum(rho_b, axis=2)
        rho_d = np.sum(rho_d, axis=2)
        skip = 24 
    samples = np.concatenate((samples_raw, np.zeros((sh[0], sh[1], 1))), axis=2).copy()
    samples = samples[:, :, skip:skip+tot]
    w0 = samples_raw[:, :, skip+3:skip+3+dim]
    log_sigma_w = samples_raw[:, :, skip+3+dim:skip+3+2*dim]
    a = samples_raw[:, :, skip+3+2*dim:skip+3+3*dim]

    samples[:, :, 2] = samples[:, :, 2]/1E-3 # R
    samples[:, :, 3] = np.average(w0, axis=2, weights=a) # w0
    if flatten:
        sigma_w = np.exp(log_sigma_w)
        samples[:, :, 0] = np.exp(samples[:, :, 0])/1E-5 # nu0
        samples[:, :, 4] = np.average(sigma_w, axis=2, weights=a) # sigma_w
    else:
        samples[:, :, 4] = np.average(log_sigma_w, axis=2, weights=a) # sigma_w
    if not simple:
        # shift
        for i in range(tot-1,1, -1):
            samples[:, :, i] = samples[:, :, i-2]
        samples[:, :, 1] = rho_d/1E-2
        samples[:, :, 0] = rho_b/1E-2
    if flatten:
        samples = samples.reshape(-1, tot)
    return samples, labels

# ========================================================================
# ========================================================================

def plot_fitting_no(sampler, data, dim, alpha=0.01, n=200, log=False, path=None, simple=True, dpi=150):
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
            rhoDM=0,
            sigmaDD=0,
            hDD=1,
            nu0=np.exp(flat_samples[index, skip+0]),
            zsun=flat_samples[index, skip+1],
            R=flat_samples[index, skip+2],
            w0=flat_samples[index, skip+3:skip+3+dim],
            sigma_w=np.exp(flat_samples[index, skip+3+dim:skip+3+2*dim]),
            a=flat_samples[index, skip+3+2*dim:skip+3+3*dim]
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

def get_dataframe_no(sampler_new, dim, locs, scales, norm_list, uni_list, nwalkers, simple=True):
    flat_samples = sampler_new.get_chain().copy()
    log_posterior_chain = sampler_new.get_log_prob().copy()
    thetas = []
    t0 = time()
    t1 = t0
    skip = 0 if simple else 24
    delta = 0
    for i, post in enumerate(log_posterior_chain.T):
        delta  = t1-t0
        progressbar(i/nwalkers*100, 30, flush=True, delta=delta, )
        for samples, lg_posterior in zip(tqdm(flat_samples[:, i], desc=f"{i}/{nwalkers}"), post):
            theta = {
                "log_nu0": samples[skip+0],
                "zsun": samples[skip+1],
                "R": samples[skip+2],
                "w0": samples[skip+3:skip+3+dim],
                "log_sigma_w": samples[skip+3+dim:skip+3+2*dim],
                "a": samples[skip+3+2*dim:skip+3+3*dim]
            }
            if simple:
                theta['rhos'] = rhos
                theta['sigmaz'] = sigmaz
            else:
                theta['rhos'] = samples[:12]
                theta['sigmaz'] = samples[12:24]
            lg_prior = log_prior(theta, locs, scales, norm_list, uni_list)
            theta['log_prior'] = lg_prior
            theta['log_posterior'] = lg_posterior
            theta['log_likelihood'] = lg_posterior - lg_prior
            del theta['w0'], theta['log_sigma_w'], theta['a'], theta['rhos'], theta['sigmaz']
            for j in range(dim):
                theta[f"w0_{j}"] = samples[skip+3+j]
                theta[f"log_sigma_w_{j}"] = samples[skip+3+dim+j]
                theta[f"a_{j}"] = samples[skip+3+2*dim+j]
            if not simple:
                for j in range(12):
                    theta[f"rhob_{j}"] = samples[j]
                    theta[f"sigmaz_{j}"] = samples[12+j]
            theta['walker'] = i
            thetas.append(theta)
        t1 = time()
    t1 = time()
    delta = t1-t0
    progressbar(100, 30, flush=True, delta=delta)
    print(f"Time: {t1-t0:.2f} s")
    df = pd.DataFrame(thetas)
    ## save
    df_dm = vaex.from_pandas(df)
    return df_dm