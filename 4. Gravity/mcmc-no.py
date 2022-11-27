import numpy as np
import vaex
from os.path import join, abspath
from os import pardir, mkdir
import os
from scipy.interpolate import interp1d
from scipy import interpolate
from scipy.stats import norm, uniform
from scipy.integrate import quad, odeint, simps
from scipy.optimize import curve_fit
from scipy import optimize
import emcee
import sys
from multiprocessing import Pool, cpu_count
from operator import itemgetter


# import utils
util_dir = abspath(pardir)
sys.path.insert(0, util_dir)
from utils import log_nu_mod

os.environ["OMP_NUM_THREADS"] = "16"

root_data_dir = abspath(join(pardir, "Data"))
data_baryon_dir = join(root_data_dir, "Baryon")
data_number_dir = join(root_data_dir, "Number-Density")
data_velocity_dir = join(root_data_dir, "Spectral-Class-Velocity")
data_comp_dir = join(root_data_dir, "Effective-Volume")

def main():
    tipe = sys.argv[1]
    burn = int(sys.argv[2])
    step = int(sys.argv[3])
    version = sys.argv[4]

    ndim, nwalkers, priors, data, p0 = initialization(tipe)
     
    sampler_ = 0
    labels = ["Nu0", 'zsun', 'R', 'sigma_w', 'w0', 'N0']
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[priors, data], pool=pool)
        state = sampler.run_mcmc(p0, burn, progress=True)
        sampler.reset()
        sampler.run_mcmc(state, step, progress=True)
        sampler_ = sampler
        print(
          "Mean acceptance fraction: {0:.3f}".format(
              np.mean(sampler.acceptance_fraction)
          )
        )
    chain = sampler_.get_chain()
    chain.shape

    df = []
    for i in range(nwalkers):
        df_new = vaex.from_dict({
            'rho1': chain[:, i, 0],
            'rho2': chain[:, i, 1],
            'rho3': chain[:, i, 2],
            'rho4': chain[:, i, 3],
            'rho5': chain[:, i, 4],
            'rho6': chain[:, i, 5],
            'rho7': chain[:, i, 6],
            'rho8': chain[:, i, 7],
            'rho9': chain[:, i, 8],
            'rho10': chain[:, i, 9],
            'rho11': chain[:, i, 10],
            'rho12': chain[:, i, 11],
            'sigmaz1': chain[:, i, 12],
            'sigmaz2': chain[:, i, 13],
            'sigmaz3': chain[:, i, 14],
            'sigmaz4': chain[:, i, 15],
            'sigmaz5': chain[:, i, 16],
            'sigmaz6': chain[:, i, 17],
            'sigmaz7': chain[:, i, 18],
            'sigmaz8': chain[:, i, 19],
            'sigmaz9': chain[:, i, 20],
            'sigmaz10': chain[:, i, 21],
            'sigmaz11': chain[:, i, 22],
            'sigmaz12': chain[:, i, 23],
            'Nu0': chain[:, i, 24],
            'zsun': chain[:, i, 25],
            'R': chain[:, i, 26],
            'sigma_w': chain[:, i, 27],
            'w0': chain[:, i, 28],
            'N0': chain[:, i, 29],
            'walker': np.repeat(i, len(chain))
        })
        if len(df) == 0:
            df = df_new
        else:
            df = df.concat(df_new)

    data_mcmc_dir = join(root_data_dir, "MCMC")
    name = f"mcmc-{tipe}-{version}.hdf5"
    df.export(join(data_mcmc_dir, name), progress=True)
    print(name)

def log_prior(theta, locs, scales):
    args = ('rhos', 'sigmaz', 'Nu0', 'zsun', 'R', 'sigma_w', 'w0', 'N0')
    rhos, sigmaz, Nu0, zsun, R, sigma_w, w0, N0 = itemgetter(*args)(theta)
    args = ('rhos_loc', 'sigmaz_loc', 'Nu0_loc', 
            'zsun_loc', 'R_loc', 'sigma_w_loc', 'w0_loc', 'N0_loc')
    rhos_loc, sigmaz_loc, Nu0_loc, zsun_loc, R_loc, sigma_w_loc, w0_loc, N0_loc = itemgetter(*args)(locs)
    args = ('rhos_scale', 'sigmaz_scale', 'Nu0_scale',
            'zsun_scale', 'R_scale', 'sigma_w_scale', 'w0_scale', 'N0_scale')
    rhos_scale, sigmaz_scale, Nu0_scale, zsun_scale, R_scale, sigma_w_scale, w0_scale, N0_scale = itemgetter(*args)(scales)
    uni_loc = np.array([Nu0_loc, zsun_loc, sigma_w_loc, w0_loc, N0_loc])
    uni_scale = np.array([Nu0_scale, zsun_scale, sigma_w_scale, w0_scale, N0_scale])
    uni_val = Nu0, zsun, sigma_w, w0, N0
    log_uni = np.sum(uniform.logpdf(uni_val, loc=uni_loc, scale=uni_scale))
    result = (np.sum(norm.logpdf(rhos, loc=rhos_loc, scale=rhos_scale))
            +np.sum(norm.logpdf(sigmaz, loc=sigmaz_loc, scale=sigmaz_scale))
            +norm.logpdf(R, loc=R_loc, scale=R_scale)
            +log_uni)
    return result

def log_likelihood(theta, z, w):
    args = ('sigma_w', 'w0', 'N0')
    sigma_w, w0, N0 = itemgetter(*args)(theta)
    
    args = ('zs', 'znum', 'zerr')
    zs, znum, zerr = itemgetter(*args)(z)
    args = ('ws', 'wnum', 'werr')
    ws, wnum, werr = itemgetter(*args)(w)
    
    resz = np.exp(log_nu_mod(zs, theta))-znum
    resultz = np.sum(norm.logpdf(resz, loc=0, scale=zerr))
    
    resw = N0*norm.pdf(ws, loc=w0, scale=sigma_w)-wnum
    resultw = np.sum(norm.logpdf(resw, loc=0, scale=werr))
    
    return resultz+resultw
 
def log_posterior(x, priors, data):
    theta = dict(rhos=x[:12], sigmaz=x[12:24], rhoDM=0, sigmaDD=0, hDD=1, Nu0=x[24], zsun=x[25], R=x[26], sigma_w=x[27], w0=x[28], N0=x[29])
    locs, scales = itemgetter('locs', 'scales')(priors)
    z, w = itemgetter('z', 'w')(data)
    log_prior_ = log_prior(theta, locs, scales)
    if not np.isfinite(log_prior_):
        return -np.inf
    log_likelihood_ = log_likelihood(theta, z, w)
    return log_prior_ + log_likelihood_
    
def initialization(tipe):
    df_number = vaex.open(join(data_number_dir, f"cum-{tipe}.hdf5"))
    df_velocity = vaex.open(join(data_velocity_dir, f"{tipe}-type.hdf5"))
    df_comp = vaex.open(join(data_comp_dir, "comp.hdf5"))

    z = df_number.z.to_numpy()
    comp = df_comp[tipe].to_numpy()
    zs = df_comp.z.to_numpy()
    compz = interp1d(zs, comp, kind='cubic', fill_value='extrapolate')
    ze = np.linspace(-0.2, 0.2, 31)
    zs = (ze[:-1]+ze[1:])/2

    znum, _ = np.histogram(z, bins=ze)
    znum = znum/compz(zs)
    zerr = np.sqrt(znum)
    znum, zerr = znum/(ze[1]-ze[0]), zerr/(ze[1]-ze[0])

    index = []
    for i, hh in enumerate(znum):
        if hh == 0:
            index.append(i)
    znum, zerr, zs = np.delete(znum, index), np.delete(zerr, index), np.delete(zs, index)

    w = df_velocity.w.to_numpy()
    we = np.linspace(-50, 50, 51)
    ws = (we[:-1]+we[1:])/2

    wnum, _ = np.histogram(w, bins=we)
    werr = np.sqrt(wnum)
    wnum, werr = wnum/(we[1]-we[0]), werr/(we[1]-we[0])

    index = []
    for i, hh in enumerate(wnum):
        if hh == 0:
            index.append(i)
    wnum, werr, ws = np.delete(wnum, index), np.delete(werr, index), np.delete(ws, index)

    ndim = 30
    nwalkers = ndim*3

    df_baryon = vaex.open(join(data_baryon_dir, "baryon.hdf5"))
    rhos = df_baryon['rho'].to_numpy()
    e_rhos = df_baryon['e_rho'].to_numpy()
    sigmaz = df_baryon['sigma_z'].to_numpy()
    e_sigmaz = df_baryon['e_sigma_z'].to_numpy()

    #initial guess
    rhos_0 = np.random.normal(loc=rhos, scale=e_rhos, size=(nwalkers, 12))
    sigmaz_0 = np.random.normal(loc=sigmaz, scale=e_sigmaz, size=(nwalkers, 12))
    
    Nu0_loc, Nu0_scale = 50000, 100000
    Nu0_0 = np.random.uniform(low=Nu0_loc, high=Nu0_loc+Nu0_scale, size=nwalkers)

    zsun_loc, zsun_scale = -0.03, 0.06
    zsun_0 = np.random.uniform(low=zsun_loc, high=zsun_loc+zsun_scale, size=nwalkers)

    R_loc, R_scale = 3.4E-3, 0.6E-3
    R_0 = np.random.normal(loc=R_loc, scale=R_scale, size=nwalkers)

    sigma_w_loc, sigma_w_scale = 1, 15
    sigma_w_0 = np.random.uniform(low=sigma_w_loc, high=sigma_w_loc+sigma_w_scale, size=nwalkers)

    w0_loc, w0_scale = -20, 40
    w0_0 = np.random.uniform(low=w0_loc, high=w0_loc+w0_scale, size=nwalkers)

    N0_loc, N0_scale = 900, 1100
    N0_0 = np.random.uniform(low=N0_loc, high=N0_loc+N0_scale, size=nwalkers)

    p0 = np.array([*rhos_0.T, *sigmaz_0.T, Nu0_0, zsun_0, R_0, sigma_w_0, w0_0, N0_0]).T

    locs = dict(rhos_loc=rhos, sigmaz_loc=sigmaz, 
            Nu0_loc=Nu0_loc, zsun_loc=zsun_loc, R_loc=R_loc, sigma_w_loc=sigma_w_loc, w0_loc=w0_loc, N0_loc=N0_loc)
    scales = dict(rhos_scale=e_rhos, sigmaz_scale=e_sigmaz, 
                  Nu0_scale=Nu0_scale, zsun_scale=zsun_scale, R_scale=R_scale, 
                  sigma_w_scale=sigma_w_scale, w0_scale=w0_scale, N0_scale=N0_scale)

    priors = dict(locs=locs, scales=scales)
    data = dict(w=dict(wnum=wnum, ws=ws, werr=werr), z=dict(znum=znum, zs=zs, zerr=zerr))
    
    return ndim, nwalkers, priors, data, p0
if __name__ == "__main__":
    main()
