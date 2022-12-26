import numpy as np
import vaex
from os.path import join, abspath
from os import pardir, mkdir
from scipy.interpolate import interp1d
import emcee
import sys

# import utils
util_dir = abspath(pardir)
sys.path.insert(0, util_dir)
from utils import log_posterior

root_data_dir = abspath(join(pardir, "Data"))
data_baryon_dir = join(root_data_dir, "Baryon")
data_number_dir = join(root_data_dir, "Number-Density")
data_velocity_dir = join(root_data_dir, "Spectral-Class-Velocity")
data_comp_dir = join(root_data_dir, "Effective-Volume")

df_baryon = vaex.open(join(data_baryon_dir, "baryon.hdf5"))

def main():
    tipe = sys.argv[1]

    df_number = vaex.open(join(data_number_dir, f"cum-{tipe}.hdf5"))
    df_velocity = vaex.open(join(data_velocity_dir, f"{tipe}-type.hdf5"))
    df_comp = vaex.open(join(data_comp_dir, "comp.hdf5"))

    z = df_number.z.to_numpy()
    w = df_velocity.w.to_numpy()
    comp = df_comp[tipe].to_numpy()
    zz = df_comp.z.to_numpy()
    comp_z = interp1d(zz, comp, kind='cubic', fill_value='extrapolate')

    ndim = 33
    nwalkers = 300

    rhos = df_baryon['rho'].to_numpy()
    e_rhos = df_baryon['e_rho'].to_numpy()
    sigmaz = df_baryon['sigma_z'].to_numpy()
    e_sigmaz = df_baryon['e_sigma_z'].to_numpy()

    #initial guess
    rhos_0 = np.random.normal(loc=rhos, scale=e_rhos, size=(nwalkers, 12))
    sigmaz_0 = np.random.normal(loc=sigmaz, scale=e_sigmaz, size=(nwalkers, 12))

    rhoDM_loc, rhoDM_scale = 0, 0.06
    rhoDM_0 = np.random.uniform(low=rhoDM_loc, high=rhoDM_loc+rhoDM_scale, size=nwalkers)

    sigmaDD_loc, sigmaDD_scale = 0, 30
    sigmaDD_0 = np.random.uniform(low=sigmaDD_loc, high=sigmaDD_loc+sigmaDD_scale, size=nwalkers)

    hDD_loc, hDD_scale = 0, 100
    hDD_0 = np.random.uniform(low=hDD_loc, high=hDD_loc+hDD_scale, size=nwalkers)


    logNu0_loc, logNu0_scale = 10, 10
    logNu0_0 = np.random.uniform(low=logNu0_loc, high=logNu0_loc+logNu0_scale, size=nwalkers)

    zsun_loc, zsun_scale = -0.05, 0.10
    zsun_0 = np.random.uniform(low=zsun_loc, high=zsun_loc+zsun_scale, size=nwalkers)

    R_loc, R_scale = 3.4E-3, 0.6E-3
    R_0 = np.random.normal(loc=R_loc, scale=R_scale, size=nwalkers)

    sigma_range = dict(A=9, F=20, G=20)
    sigma_w_loc, sigma_w_scale = 1, sigma_range[tipe]
    sigma_w_0 = np.random.uniform(low=sigma_w_loc, high=sigma_w_loc+sigma_w_scale, size=nwalkers)

    w0_loc, w0_scale = -20, 40
    w0_0 = np.random.uniform(low=w0_loc, high=w0_loc+w0_scale, size=nwalkers)
    
    N0_range = dict(A=[50, 200], F=[1000, 1000], G=[1000,1000])
    N0_loc, N0_scale = N0_range[tipe][0], N0_range[tipe][1]
    N0_0 = np.random.uniform(low=N0_loc, high=N0_loc+N0_scale, size=nwalkers)

    p0 = np.array([*rhos_0.T, *sigmaz_0.T, rhoDM_0, sigmaDD_0, hDD_0, logNu0_0, zsun_0, R_0, sigma_w_0, w0_0, N0_0]).T

    # Save dir
    data_dir_mcmc = join(root_data_dir, "MCMC")

    filename = join(data_dir_mcmc, f"chain-{tipe}.h5")
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)

    locs = dict(rhos_loc=rhos, sigmaz_loc=sigmaz, rhoDM_loc=rhoDM_loc, sigmaDD_loc=sigmaDD_loc, hDD_loc=hDD_loc, 
                logNu0_loc=logNu0_loc, zsun_loc=zsun_loc, R_loc=R_loc, sigma_w_loc=sigma_w_loc, w0_loc=w0_loc, N0_loc=N0_loc)
    scales = dict(rhos_scale=e_rhos, sigmaz_scale=e_sigmaz, rhoDM_scale=rhoDM_scale, sigmaDD_scale=sigmaDD_scale, 
                  hDD_scale=hDD_scale, logNu0_scale=logNu0_scale, zsun_scale=zsun_scale, R_scale=R_scale, 
                  sigma_w_scale=sigma_w_scale, w0_scale=w0_scale, N0_scale=N0_scale)
    priors = dict(locs=locs, scales=scales)

    df_number = vaex.open(join(data_number_dir, f"cum-{tipe}.hdf5"))
    df_velocity = vaex.open(join(data_velocity_dir, f"{tipe}-type.hdf5"))
    df_comp = vaex.open(join(data_comp_dir, "comp.hdf5"))

    z = df_number.z.to_numpy()
    w = df_velocity.w.to_numpy()
    comp = df_comp[tipe].to_numpy()
    zz = df_comp.z.to_numpy()
    comp_z = interp1d(zz, comp, kind='cubic', fill_value='extrapolate')
    data = dict(z=z, w=w)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[priors, data, comp_z], backend=backend)

    state = sampler.run_mcmc(p0, 1000, progress=True)
    run2_backend = emcee.backends.HDFBackend(filename, name="mcmc_second_prior")
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[priors, data, comp_z], backend=run2_backend)
    sampler.reset()
    sampler.run_mcmc(state, 5000, progress=True)
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
