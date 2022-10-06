import vaex
import numpy as np
from operator import itemgetter
from scipy.interpolate import interp1d
from scipy.stats import norm, uniform
from scipy.integrate import simps, quad
import emcee
from os.path import join, abspath
from os import pardir, mkdir
import sys
# import utils
util_dir = abspath(pardir)
sys.path.insert(0, util_dir)
from utils import log_posterior, log_nu_mod

root_data_dir = abspath(join(pardir, "Data"))
data_baryon_dir = join(root_data_dir, "Baryon")
data_mcmc_dir = join(root_data_dir, "MCMC")
df_baryon = vaex.open(join(data_baryon_dir, "baryon.hdf5"))

def cumulative(z, theta):
    nu_f = lambda zz: np.exp(log_nu_mod(zz, theta))/theta["Nu0"]
    zz = np.linspace(-1,1, 1000)
    integral = simps(nu_f(zz), zz)
    zz = np.array([np.linspace(-1, zs, 1000) for zs in z])
    result = np.array([simps(nu_f(zs), zs) for zs in zz])/integral
    return result

def main():
  rhos = df_baryon['rho'].to_numpy()
  e_rhos = df_baryon['e_rho'].to_numpy()
  sigmaz = df_baryon['sigma_z'].to_numpy()
  e_sigmaz = df_baryon['e_sigma_z'].to_numpy()

  # test the following dark values
  rhoDM = 0.016 # Msun/pc^3
  sigmaDD = 7 # Msun/pc^2
  hDD = 20 # pc
  Nu0 = 12000 # pc^-1
  zsun = 0.0001 # kpc
  sigma_w = 5 # km/s
  w0 = -7 # km/s
  R = 3.4E-3 # Msun/pc^3
  N0 = 200 # (km/s)^-1
  theta = dict(rhos=rhos, sigmaz=sigmaz, rhoDM=rhoDM, sigmaDD=sigmaDD, hDD=hDD, Nu0=Nu0, zsun=zsun, R=R, sigma_w=sigma_w, w0=w0, N0=N0)

  zz = np.linspace(-0.2, 0.2, 50)
  compz = lambda x: -0.05*norm.pdf(x, loc=0, scale=0.04)+0.8

  # mock stars
  nu_f = lambda zz: np.exp(log_nu_mod(zz, theta))/theta["Nu0"]*compz(zz)
  integral_nu, _ = quad(nu_f, -1, 1)
  invers = interp1d(cumulative(zz, theta), zz, kind="cubic", fill_value="extrapolate")
  Num = int(Nu0*integral_nu)
  rand = np.random.random(Num)
  z_num = invers(rand)
  w_num = norm.rvs(loc=w0, scale=sigma_w, size=N0)
  
  # initial guesses
  ndim = 33
  nwalkers = ndim*3

  rhos_0 = np.random.normal(loc=rhos, scale=e_rhos, size=(nwalkers, 12))
  sigmaz_0 = np.random.normal(loc=sigmaz, scale=e_sigmaz, size=(nwalkers, 12))

  rhoDM_loc, rhoDM_scale = 0, 0.06
  rhoDM_0 = np.random.uniform(low=rhoDM_loc, high=rhoDM_loc+rhoDM_scale, size=nwalkers)

  sigmaDD_loc, sigmaDD_scale = 0, 30
  sigmaDD_0 = np.random.uniform(low=sigmaDD_loc, high=sigmaDD_loc+sigmaDD_scale, size=nwalkers)

  hDD_loc, hDD_scale = 0, 100
  hDD_0 = np.random.uniform(low=hDD_loc, high=hDD_loc+hDD_scale, size=nwalkers)


  Nu0_loc, Nu0_scale = 10000, 4000
  Nu0_0 = np.random.uniform(low=Nu0_loc, high=Nu0_loc+Nu0_scale, size=nwalkers)

  zsun_loc, zsun_scale = -0.05, 0.10
  zsun_0 = np.random.uniform(low=zsun_loc, high=zsun_loc+zsun_scale, size=nwalkers)

  R_loc, R_scale = 3.4E-3, 0.6E-3
  R_0 = np.random.normal(loc=R_loc, scale=R_scale, size=nwalkers)

  # sigma_range = dict(A=9, F=20, G=20)
  sigma_w_loc, sigma_w_scale = 1, 10
  sigma_w_0 = np.random.uniform(low=sigma_w_loc, high=sigma_w_loc+sigma_w_scale, size=nwalkers)

  w0_loc, w0_scale = -25, 50
  w0_0 = np.random.uniform(low=w0_loc, high=w0_loc+w0_scale, size=nwalkers)

  N0_loc, N0_scale = 150, 250
  N0_0 = np.random.uniform(low=N0_loc, high=N0_loc+N0_scale, size=nwalkers)

  p0 = np.array([*rhos_0.T, *sigmaz_0.T, rhoDM_0, sigmaDD_0, hDD_0, Nu0_0, zsun_0, R_0, sigma_w_0, w0_0, N0_0]).T

  # MCMC
  locs = dict(rhos_loc=rhos, sigmaz_loc=sigmaz, rhoDM_loc=rhoDM_loc, sigmaDD_loc=sigmaDD_loc, hDD_loc=hDD_loc, 
              Nu0_loc=Nu0_loc, zsun_loc=zsun_loc, R_loc=R_loc, sigma_w_loc=sigma_w_loc, w0_loc=w0_loc, N0_loc=N0_loc)
  scales = dict(rhos_scale=e_rhos, sigmaz_scale=e_sigmaz, rhoDM_scale=rhoDM_scale, sigmaDD_scale=sigmaDD_scale, 
                hDD_scale=hDD_scale, Nu0_scale=Nu0_scale, zsun_scale=zsun_scale, R_scale=R_scale, 
                sigma_w_scale=sigma_w_scale, w0_scale=w0_scale, N0_scale=N0_scale)
  priors = dict(locs=locs, scales=scales)

  data = dict(z=z_num, w=w_num)

  sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[priors, data, compz])
  state = sampler.run_mcmc(p0, 5000, progress=True)
  sampler.reset()
  sampler.run_mcmc(state, 30000, progress=True)
  # save to dataframe
  flat_samples = sampler.get_chain(flat=True)
  dict_mcmc = {
      "rho1": flat_samples[:, 0],
      "rho2": flat_samples[:, 1],
      "rho3": flat_samples[:, 2],
      "rho4": flat_samples[:, 3],
      "rho5": flat_samples[:, 4],
      "rho6": flat_samples[:, 5],
      "rho7": flat_samples[:, 6],
      "rho8": flat_samples[:, 7],
      "rho9": flat_samples[:, 8],
      "rho10": flat_samples[:, 9],
      "rho11": flat_samples[:, 10],
      "rho12": flat_samples[:, 11],
      "sigmaz1": flat_samples[:, 12],
      "sigmaz2": flat_samples[:, 13],
      "sigmaz3": flat_samples[:, 14],
      "sigmaz4": flat_samples[:, 15],
      "sigmaz5": flat_samples[:, 16],
      "sigmaz6": flat_samples[:, 17],
      "sigmaz7": flat_samples[:, 18],
      "sigmaz8": flat_samples[:, 19],
      "sigmaz9": flat_samples[:, 20],
      "sigmaz10": flat_samples[:, 21],
      "sigmaz11": flat_samples[:, 22],
      "sigmaz12": flat_samples[:, 23],
      "rhob": np.sum(flat_samples[:, 0:12], axis=1),
      "rhoDM": flat_samples[:, 24],
      "sigmaDD": flat_samples[:, 25],
      "hDD": flat_samples[:, 26], 
      "logNu0": flat_samples[:, 27], 
      "zsun": flat_samples[:, 28], 
      "R": flat_samples[:, 29], 
      "sigma_w": flat_samples[:, 30], 
      "w0": flat_samples[:, 31], 
      "N0": flat_samples[:, 32]
  }
  df_mcmc = vaex.from_dict(dict_mcmc)
  df_mcmc.export(join(data_mcmc_dir, "model.hdf5"), progress=True)
  
if __name__ == "__main__":
  main()