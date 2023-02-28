import sys
from os.path import join, abspath, pardir
from glob import glob
import numpy as np
import vaex
from tqdm import tqdm

root_dir = abspath(pardir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from utils import load_data, plot_fitting_dm, plot_corner_dm, run_mcmc, plot_data, initialize_prior_dm, initialize_walkers_dm
from utils import safe_mkdir, log_posterior_simple_DM, log_prior

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

number_files = glob(join(data_eff_dir, "*.hdf5"))
number_files.sort()
velocity_files = glob(join(data_velocity_dir, "gaia_*.hdf5"))
velocity_files.sort()
velocity_popt_files = glob(join(data_velocity_dir, "popt_gaia_*.hdf5"))
velocity_popt_files.sort()

# main program
data, dim, w0, sigma_w, a_raw, name = load_data(index, number_files, velocity_files, velocity_popt_files)

asset_dir = join(data_dir, f"assets-{name}")
safe_mkdir(asset_dir)

plot_data(data, path=join(asset_dir, "data.png"))

locs, scales, uni_list_DM, norm_list_DM = initialize_prior_dm(w0, sigma_w, a_raw, dim)

p0, ndim, nwalkers = initialize_walkers_dm(locs, scales, dim)

labels = [r"$\log \rho_{DM}$",r"$\log \nu_0$ " , r"$z_{\odot}$", r"$R$"]

sampler = run_mcmc(
    nwalkers, ndim, p0, labels, log_posterior_simple_DM,
    args=[data, locs, scales, dim, norm_list_DM, uni_list_DM], 
    cores=16, plot=True, step=1000, path=join(asset_dir, "chain0.png"))

next_p0 = sampler.get_chain()[-1]
# next_p0 = sampler_new.get_chain()[-1]
sampler_new = run_mcmc(
    nwalkers, ndim, next_p0, labels, log_posterior_simple_DM,
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
        "log_rhoDM": samples[0],
        "log_nu0": samples[1],
        "zsun": samples[2],
        "R": samples[3],
        "w0": samples[4:4+dim],
        "log_sigma_w": samples[4+dim:4+2*dim],
        "a": samples[4+2*dim:4+3*dim]
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
df_dm["log_likelihood"] = log_likelihood
df_dm["log_prior"] = log_priors
df_dm["log_posterior"] = log_posterior
df_dm = vaex.from_dict(df_dm)

df_dm.export(join(data_dir, f"dm_simple_{name}"), progress=True)