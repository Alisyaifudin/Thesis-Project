import numpy as np
import vaex
from os.path import join, abspath
from os import pardir, mkdir
import emcee
import sys

# import utils
util_dir = abspath(pardir)
sys.path.insert(0, util_dir)
from utils import log_posterior

def main():
  tipe = input("Spectral class: (A/F/G) ")
  
  root_data_dir = abspath(join(pardir, "Data"))
  # Initialization
  data_baryon_dir = join(root_data_dir, "Baryon")
  df = vaex.open(join(data_baryon_dir, "baryon.hdf5"))
  rhos = df["rho"].to_numpy()  # Msun/pc^3
  rhos_std = df["e_rho"].to_numpy()
  sigmaz = df["sigma_z"].to_numpy() # km/s
  sigmaz_std = df["e_sigma_z"].to_numpy()
  ndim = 24+5 # 29
  nwalkers = 100
  rhos_0 = np.random.normal(loc=rhos, scale=rhos_std, size=(nwalkers, 12))
  sigmaz_0 = np.random.normal(loc=sigmaz, scale=sigmaz_std, size=(nwalkers, 12))
  rhoDM_loc, rhoDM_scale = 0, 0.06
  rhoDM_0 = np.random.uniform(low=rhoDM_loc, high=rhoDM_loc+rhoDM_scale, size=nwalkers)
  sigmaDD_loc, sigmaDD_scale = 0, 30
  sigmaDD_0 = np.random.uniform(low=sigmaDD_loc, high=sigmaDD_loc+sigmaDD_scale, size=nwalkers)
  hDD_loc, hDD_scale = 0, 100
  hDD_0 = np.random.uniform(low=hDD_loc, high=hDD_loc+hDD_scale, size=nwalkers)
  loc_nv = dict(A=12, F=14, G=13.8)
  scale_nv = dict(A=1, F=0.4, G=0.2)
  Nv_loc, Nv_scale = loc_nv[tipe], scale_nv[tipe]
  Nv_0 = np.random.uniform(low=Nv_loc, high=Nv_loc+Nv_scale, size=nwalkers)
  zsun_loc, zsun_scale = -0.05, 0.10
  zsun_0 = np.random.uniform(low=zsun_loc, high=zsun_loc+zsun_scale, size=nwalkers)

  p0 = np.array([*rhos_0.T, *sigmaz_0.T, rhoDM_0, sigmaDD_0, hDD_0, Nv_0, zsun_0]).T

  # Save dir
  data_dir_mcmc = join(root_data_dir, "MCMC")

  filename = join(data_dir_mcmc, f"chain-{tipe}.h5")
  backend = emcee.backends.HDFBackend(filename)
  backend.reset(nwalkers, ndim)
  locs = [*rhos, *sigmaz, rhoDM_loc, sigmaDD_loc, hDD_loc, Nv_loc, zsun_loc]
  scales = [*rhos_std, *sigmaz_std, rhoDM_scale, sigmaDD_scale, hDD_scale, Nv_scale, zsun_scale]
  priors = dict(locs=locs, scales=scales)
  sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[priors, tipe], backend=backend)

  state = sampler.run_mcmc(p0, 1000, progress=True)
  run2_backend = emcee.backends.HDFBackend(filename, name="mcmc_second_prior")
  sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[priors, tipe], backend=run2_backend)
  sampler.reset()
  sampler.run_mcmc(state, 10000, progress=True)
  print(
    "Mean acceptance fraction: {0:.3f}".format(
        np.mean(sampler.acceptance_fraction)
    )
  )
  print(
      "Mean autocorrelation time: {0:.3f} steps".format(
          np.mean(sampler.get_autocorr_time())
      )
  )


if __name__ == "__main__":
  main()