import sys
from os.path import join, abspath, pardir
from glob import glob
import numpy as np
import vaex
from tqdm import tqdm
import corner
from matplotlib import pyplot as plt

root_dir = abspath(pardir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from utils import (load_data, plot_data, run_mcmc, nu_mod, fw, 
                    log_prior, log_posterior_dm, safe_mkdir)

# input from user
index = int(sys.argv[1])
step = int(sys.argv[2])
# =================

root_data_dir = abspath(join(root_dir, "Data"))
data_dir = join(root_data_dir, "MCMC")
data_baryon_dir = join(root_data_dir, "Baryon")
data_velocity_dir = join(root_data_dir, "Velocity-Distribution")
data_eff_dir = join(root_data_dir, "Effective-Volume")
# load baryons components
df_baryon = vaex.open(join(data_baryon_dir, "baryon.hdf5"))
rhos = df_baryon["rho"].to_numpy()  # Msun/pc^3
sigmaz = df_baryon["sigma_z"].to_numpy() # km/s
e_rhos = df_baryon["e_rho"].to_numpy()  # Msun/pc^3
e_sigmaz = df_baryon["e_sigma_z"].to_numpy() # km/s

# load files
number_files = glob(join(data_eff_dir, "*.hdf5"))
number_files.sort()
velocity_files = glob(join(data_velocity_dir, "gaia_*.hdf5"))
velocity_files.sort()
velocity_popt_files = glob(join(data_velocity_dir, "popt_gaia_*.hdf5"))
velocity_popt_files.sort()
# additional functions
def initialize_prior_dm(rhos, sigmaz, e_rhos, e_sigmaz, w0, sigma_w, a_raw, dim):
    locs = dict(
        rhos=rhos,
        sigmaz=sigmaz,
        log_rhoDM=np.log(0.001), 
        log_nu0=np.log(1E-5), 
        zsun=-20, 
        R=3.4E-3, 
        w0=w0-5, 
        log_sigma_w=np.log(sigma_w*0.7), 
        a=np.select([a_raw > 0], [a_raw*0.7], default=a_raw*1.3)
    )
    
    scales = dict(
        rhos=e_rhos,
        sigmaz=e_sigmaz,
        log_rhoDM=np.log(0.1)-locs['log_rhoDM'], 
        log_nu0=np.log(1E-4)-locs['log_nu0'], 
        zsun=20-locs['zsun'], 
        R=0.6E-3, 
        w0=np.repeat(10, dim), 
        log_sigma_w=np.log(sigma_w*1.3)-locs['log_sigma_w'], 
        a=np.abs(a_raw*0.6)
    )
    uni_list_DM = ['log_rhoDM', 'log_nu0', 'zsun', 'log_sigma_w', 'w0', 'a']
    norm_list_DM = ['R', 'rhos', 'sigmaz']
    return locs, scales, uni_list_DM, norm_list_DM

def initialize_walkers_dm(locs, scales, dim):
    theta = np.concatenate([np.ravel(x) for x in locs.values()])
    ndim = len(theta)
    nwalkers = ndim*2+1

    rhos_0 = np.random.normal(loc=locs['rhos'], scale=scales['rhos'], size=(nwalkers, 12))
    sigmaz_0 = np.random.normal(loc=locs['sigmaz'], scale=scales['sigmaz'], size=(nwalkers, 12))
    log_rhoDM_0 = np.random.uniform(low=locs['log_rhoDM'], high=locs['log_rhoDM']+scales['log_rhoDM'], size=nwalkers)
    log_nu0_0 = np.random.uniform(low=locs['log_nu0'], high=locs['log_nu0']+scales['log_nu0'], size=nwalkers)
    zsun_0 = np.random.uniform(low=locs['zsun'], high=locs['zsun']+scales['zsun'], size=nwalkers)
    R_0 = np.random.normal(loc=locs['R'], scale=scales['R'], size=nwalkers)
    w0_0 = np.random.uniform(low=locs['w0'], high=locs['w0']+scales['w0'], size=(nwalkers, dim))
    log_sigma_w_0 = np.random.uniform(low=locs['log_sigma_w'], high=locs['log_sigma_w']+scales['log_sigma_w'], size=(nwalkers, dim))
    a_0 = np.random.uniform(low=locs['a'], high=locs['a']+scales['a'], size=(nwalkers, dim))

    p0 = np.array([*rhos_0.T, *sigmaz_0.T, log_rhoDM_0, log_nu0_0, zsun_0, R_0, *w0_0.T, *log_sigma_w_0.T, *a_0.T]).T
    return p0, ndim, nwalkers

def plot_corner_dm(sampler, path=None):
    flat_samples = sampler.get_chain(flat=True).copy()
    flat_samples[:, 24+0] = np.exp(flat_samples[:, 24+0])/1E-2
    flat_samples[:, 24+1] = np.exp(flat_samples[:, 24+1])/1E-5
    flat_samples[:, 24+3] = flat_samples[:, 24+3]/1E-3
    flat_samples[:, 24+4] = np.sum(flat_samples[:, :12], axis=1)/1E-2
    flat_samples = flat_samples[:, 24:24+5]
    labels = [r"$\rho_{DM}\times 10^2$",r"$\nu_0 \times10^5$ " , r"$z_{\odot}$", r"$R\times 10^3$", r"$\rho_{b}\times 10^2$"]
    fig = corner.corner(
        flat_samples, labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True, title_kwargs={"fontsize": 12},
    )
    if path is not None:
        fig.savefig(path, dpi=300)

def plot_fitting_dm(sampler, data, dim, alpha=0.01, n=200, path=None):
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
            rhos=flat_samples[index, :12], 
            sigmaz=flat_samples[index, 12:24], 
            rhoDM=np.exp(flat_samples[index, 24]),
            sigmaDD=0, 
            hDD=1, 
            nu0=np.exp(flat_samples[index, 25]),
            zsun=flat_samples[index, 26],
            R=flat_samples[index, 27],
            w0=flat_samples[index, 28:28+dim],
            sigma_w=np.exp(flat_samples[index, 28+dim:28+2*dim]),
            a=flat_samples[index, 28+2*dim:28+3*dim]
        )
        nu = nu_mod(zs, **theta_dict)
        axes[0].plot(zs, nu, label='model', c="r", alpha=alpha)
        Fw = fw(ws, **theta_dict)
        axes[1].plot(ws, Fw, label='model', c="r", alpha=alpha)
    if path is not None:
        fig.savefig(path, dpi=300)
    plt.show()
# main program
data, dim, w0, sigma_w, a_raw, name = load_data(index, number_files, velocity_files, velocity_popt_files)

asset_dir = join(data_dir, f"assets-full-{name}")
safe_mkdir(asset_dir)

plot_data(data, path=join(asset_dir, "data.png"))

locs, scales, uni_list_DM, norm_list_DM = initialize_prior_dm(rhos, sigmaz, e_rhos, e_sigmaz, w0, sigma_w, a_raw, dim)

p0, ndim, nwalkers = initialize_walkers_dm(locs, scales, dim)

labels = [r"$\log \rho_{DM}$",r"$\log \nu_0$ " , r"$z_{\odot}$", r"$R$"]

sampler = run_mcmc(
    nwalkers, ndim, p0, labels, log_posterior_dm,
    args=[data, locs, scales, dim, norm_list_DM, uni_list_DM], 
    cores=16, plot=True, skip=24, step=1000, path=join(asset_dir, "chain0.png"))

next_p0 = sampler.get_chain()[-1]
# next_p0 = sampler_new.get_chain()[-1]
sampler_new = run_mcmc(
    nwalkers, ndim, next_p0, labels, log_posterior_dm,
    args=[data, locs, scales, dim, norm_list_DM, uni_list_DM], 
    cores=16, plot=True, step=step, path=join(asset_dir, "chain1.png"))


plot_corner_dm(sampler_new, path=join(asset_dir, "corner.png"))
plot_fitting_dm(sampler_new, data, dim, rhos, sigmaz, alpha=0.05, n=100, path=join(asset_dir, "fitting.png"))

flat_samples = sampler.get_chain(flat=True).copy()
log_posterior = sampler.get_log_prob(flat=True).copy()
# save priors
log_priors = []
for samples in tqdm(flat_samples):
    theta = {
        "rhos": samples[:12],
        "sigmaz": samples[12:24],
        "log_rhoDM": samples[24],
        "log_nu0": samples[25],
        "zsun": samples[26],
        "R": samples[27],
        "w0": samples[28:28+dim],
        "log_sigma_w": samples[28+dim:28+2*dim],
        "a": samples[28+2*dim:28+3*dim]
    }
    lg_prior = log_prior(theta, locs, scales, norm_list_DM, uni_list_DM)
    log_priors.append(lg_prior)

log_priors = np.array(log_priors)
log_likelihood = log_posterior - log_priors

flat_samples = sampler.get_chain(flat=True).copy()
df_dm = {
    "log_rhoDM": flat_samples[:, 0],
    "log_nu0": flat_samples[:, 1],
    "zsun": flat_samples[:, 2],
    "R": flat_samples[:, 3],
}
for i in range(dim):
    df_dm[f"w0_{i}"] = flat_samples[:, 4+i]
    df_dm[f"log_sigma_w_{i}"] = flat_samples[:, 4+dim+i]
    df_dm[f"a_{i}"] = flat_samples[:, 4+2*dim+i]
for i in range(12):
    df_dm[f"rhos_{i}"] = flat_samples[:, i]
    df_dm[f"sigmaz_{i}"] = flat_samples[:, 12+i]
    
df_dm["log_likelihood"] = log_likelihood
df_dm["log_prior"] = log_priors
df_dm["log_posterior"] = log_posterior
df_dm = vaex.from_dict(df_dm)

df_dm.export(join(data_dir, f"dm_{name}"), progress=True)