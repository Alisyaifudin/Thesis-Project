import numpy as np
from glob import glob
# import pathlib
import vaex
from time import time
from os.path import join
from scipy.stats import median_abs_deviation as mad
from scipy.integrate import simps
from .plot_mcmc import calculate_probs
from .style import style
from datetime import datetime

style()
# init values
locs_raw = dict(
    mu0=0.1,
    rhoDM=-0.05,
    sigmaDD=0.1,
    log_hDD=np.log(1),
    log_nu0=-1,
    R=3.4E-3,
    zsun=-50,
    w0=-15,
    log_sigmaw1=np.log(1),
    log_a1=np.log(0.05),
    log_sigmaw2=np.log(1),
    log_a2=np.log(0.05),
)

scales_raw = dict(
    mu0=5.9,
    rhoDM=0.15,
    sigmaDD=30,
    log_hDD=np.log(100)-locs_raw['log_hDD'],
    log_nu0=3,
    R=0.6E-3,
    zsun=100,
    w0=15,
    log_sigmaw1=np.log(30)-locs_raw['log_sigmaw1'],
    log_a1=np.log(2)-locs_raw['log_a1'],
    log_sigmaw2=np.log(30)-locs_raw['log_sigmaw2'],
    log_a2=np.log(2)-locs_raw['log_a2'],
)


def get_mul(flat_raw, level=0.95):
    mad_v = mad(flat_raw)
    median_v = np.median(flat_raw)

    num = []
    mul = np.linspace(1, 5, 50)
    total = len(flat_raw)
    for i in mul:
        mask = np.abs(flat_raw - median_v) < i*mad_v
        num.append(len(flat_raw[mask]))
    num = np.array(num)
    p = np.polyfit(mul, num/total, 4)
    y = np.polyval(p, mul)
    areas = []
    for i in range(len(y)-1):
        areas.append(simps(y[:i+1], mul[:i+1]))
    areas = np.array(areas)
    max_areas = areas.max()
    areas /= max_areas
    m = mul[np.argmin(np.abs(areas - level))]
    return m

def get_data(zpath, wpath, index):
    print("reading data")
    zfiles = glob(join(zpath, 'z*.hdf5'))
    zfiles.sort()
    wfiles = glob(join(wpath, 'w*.hdf5'))
    wfiles.sort()

    zfile = zfiles[index]
    wfile = wfiles[index]
    zdata = vaex.open(zfile)

    zmid = zdata['zmid'].to_numpy()
    znum = zdata['znum'].to_numpy()
    zerr = zdata['zerr'].to_numpy()

    wdata = vaex.open(wfile)
    wmid = wdata['wmid'].to_numpy()
    wnum = wdata['wnum'].to_numpy()
    werr = wdata['werr'].to_numpy()

    zdata = (zmid, znum, zerr)
    wdata = (wmid, wnum, werr)
    return zdata, wdata

def get_params(chain, indexes, labs):
    params = []
    dic = {key: value for key, value in zip(labs, indexes)}
    for l, i in dic.items():
        if l == 'rhob':
            params.append(chain[:, :, :i].sum(axis=2).T/1E-2)
        elif l == 'rhoDM':
            params.append(chain[:, :, i].T/1E-2)
        elif l == 'R':
            params.append(chain[:, :, i].T/1E-3)
        else:
            params.append(chain[:, :, i].T)
    params = np.array(params).T
    return params
  
def get_initial_position(labs, chain=None, indexes=None):
    locs = {key: value for key, value in locs_raw.items() if key in labs}
    scales = {key: value for key, value in scales_raw.items() if key in labs}
    if chain is not None:
        for k, i in zip(labs, indexes):
            if k not in ["log_sigmaw1", "log_sigmaw2", "log_a1", "log_a2"]:
                continue 
            v = chain[:, :, i]
            flat_raw = v.reshape(-1)
            mad_v = mad(flat_raw)
            median_v = np.median(flat_raw)
            m = get_mul(flat_raw, 0.9) 
            low = median_v - m*mad_v
            delta = 2*m*mad_v
            locs[k] = low
            scales[k] = delta
    locs = np.array(list(locs.values()))
    scales = np.array(list(scales.values()))
    return locs, scales

def run_mcmc(func, labs, indexes, data, output_path, steps0=1000, steps=2000, burn=500, model=2):
    # get data
    zdata, wdata = data
    # initial position
    print("Generating initial position")
    locs, scales = get_initial_position(labs)
    ndim = len(locs)+24
    nwalkers = 2*ndim+2
    p0 = func.generate_p0(nwalkers, locs, scales, kind=model)
    # run mcmc
    print(f"Running first MCMC, {steps0} steps")
    t0 = time()
    chain = func.mcmc(steps0, nwalkers, p0, zdata, wdata, locs,
                      scales, dz=1, verbose=True, parallel=True)
    print(time() - t0, "s")
    print(f"Running second MCMC {steps0} steps")
    # get second initial position
    locs, scales = get_initial_position(labs, chain[burn:, :, :], indexes)
    p0 = func.generate_p0(nwalkers, locs, scales, kind=model)
    t0 = time()
    chain = func.mcmc(steps0, nwalkers, p0, zdata, wdata, locs,
                      scales, dz=1, verbose=True, parallel=True)
    print(time() - t0, "s")
    # long mcmc
    print(f"Running long MCMC, {steps} steps")
    p0 = chain[-1, :, :]
    t0 = time()
    chain = func.mcmc(steps, nwalkers, p0, zdata, wdata, locs,
                      scales, dz=1, verbose=True, parallel=True)
    print(time() - t0, "s")
    np.save(output_path, chain)


def run_calculate_bic_aic(func, labs, data, index, chain, output_file, model=2):
    ndim = chain.shape[2]
    zdata, wdata = data
    locs = {key: value for key, value in locs_raw.items() if key in labs}
    scales = {key: value for key, value in scales_raw.items() if key in labs}
    locs = np.array(list(locs.values()))
    scales = np.array(list(scales.values()))
    # calculate likelihood
    print("Calculating likelihood")
    probs = calculate_probs(func, chain, ndim, zdata, wdata,
                            locs, scales, batch=10000)
    likelihood = probs[:, 1]
    # remove nan from likelihood
    likelihood = likelihood[~np.isnan(likelihood)]
    max_likelihood = np.max(likelihood)
    # calculate BIC
    zmid = zdata[0]
    wmid = wdata[0]
    bic = -2 * max_likelihood + ndim * np.log(3*len(zmid)+3*len(wmid))
    aic = -2 * max_likelihood + 2 * ndim
    print(f"BIC: {bic}")
    print(f"AIC: {aic}")
    with open(output_file, 'a') as f:
        f.write(f"{model},{index},{bic},{aic},{datetime.now()}\n")
