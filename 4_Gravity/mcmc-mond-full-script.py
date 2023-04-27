import sys
from os.path import join, abspath, pardir
from glob import glob
import numpy as np
import vaex
import matplotlib.pyplot as plt


# ================================================
root_dir = abspath(pardir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from utils import safe_mkdir, nu_mond, fw_mond, style
from utils import (load_data, plot_data, initialize_prior_mond,
                   initialize_walkers_mond, run_mcmc, log_posterior_mond,
                   consume_samples_mond, plot_corner, plot_fitting_mond,
                   get_dataframe_mond, inv_interpolation_simple, inv_interpolation_standard)
# ================================================
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
simple = False if sys.argv[4] == "False" else True
full = "full" if not simple else "simple"
# ************************************************
# ************************************************
# ************************************************
# dir paths
root_data_dir = abspath(join(root_dir, "Data"))
data_dir = join(root_data_dir, "MCMC-mond")
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
asset_dir = join(data_dir, f"assets-mond-{full}-"+name[:-5])
safe_mkdir(asset_dir)
print(asset_dir)
# ================================================
# ================================================
# ================================================
# load data
data, dim, w0, sigma_w, a_raw, name = load_data(index, number_files, velocity_files, velocity_popt_files)
print(name)
## plot data
plot_data(data, path=join(asset_dir, "data-plot.png"), dpi=dpi)
## initialize prior
locs, scales, uni_list, norm_list = initialize_prior_mond(dim, w0, sigma_w, a_raw, simple=simple)
## initialize walkers
p0, ndim, nwalkers = initialize_walkers_mond(locs, scales, dim, simple=simple)
sampler = run_mcmc(
    nwalkers, ndim, p0, dim, log_posterior_mond, consume_samples_mond, cores=cores,
    args=[data, locs, scales, dim, norm_list, uni_list, simple], 
    plot=True, step=1000, simple=simple, path=join(asset_dir, "mcmc-plot-0.png")
)
## run again, with the last sample as the initial point
next_p0 = sampler.get_chain()[-1]
sampler_new = run_mcmc(
    nwalkers, ndim, next_p0, dim, log_posterior_mond, consume_samples_mond, cores=cores,
    args=[data, locs, scales, dim, norm_list, uni_list, simple], 
    plot=True, step=step, simple=simple, path=join(asset_dir, "mcmc-plot-1.png")
)
# corner plot
samples = sampler_new.get_chain()
plot_corner(samples, dim, consume_samples_mond, simple=simple, path=join(asset_dir, "corner-plot.png"))
# fitting plot
plot_fitting_mond(sampler_new, data, dim, alpha=0.01, n=500, simple=simple, path=join(asset_dir, "fitting-plot.png"))
# get dataframe
df = get_dataframe_mond(sampler_new, dim, locs, scales, norm_list, uni_list, nwalkers, simple=simple)
# plot a0
log_mu0 = df['log_mu0'].to_numpy()
mu0 = np.exp(log_mu0)
a0_simple = inv_interpolation_simple(mu0)
a0_standard = inv_interpolation_standard(mu0)
plt.figure(figsize=(10, 5))
plt.hist(a0_simple, bins=300, label="simple", alpha=0.5, density=True)
plt.hist(a0_standard, bins=100, label="standard", alpha=0.5, density=True)
plt.legend()
plt.xlim(0, 5)
plt.savefig(join(asset_dir, "a0-plot.png"), dpi=dpi)
plt.show()
# save
df.export(join(data_dir, f"mond_{full}_{name}"), progress=True)