import numpy as np
from glob import glob
# import pathlib
import vaex
from time import time
from os.path import join, abspath
from scipy.stats import median_abs_deviation as mad
from scipy.integrate import simps
from .plot_mcmc import calculate_probs
from .style import style
from datetime import datetime
import pathlib

current = pathlib.Path(__file__).parent.resolve()
root_dir = join(current, "..")
root_data_dir = join(root_dir, 'Data')
baryon_dir = join(root_data_dir, "Baryon")
df_baryon = vaex.open(join(baryon_dir, "baryon.hdf5"))
style()
#init dynamics
#init dynamics
rhob = df_baryon["rho"].to_numpy()  # Msun/pc^3
sigmaz = df_baryon["sigma_z"].to_numpy() # km/s
rhob_e = df_baryon["e_rho"].to_numpy()  # Msun/pc^3
sigmaz_e = df_baryon["e_sigma_z"].to_numpy() # km/s
rhob_init = {'mean_arr': rhob, 'sigma_arr': rhob_e, 'label': r'$\rho_{b}$', 'lab': 'rhob'}
sigmaz_init = {'mean_arr': sigmaz, 'sigma_arr': sigmaz_e}
rhoDM_init = {'low': -0.05, 'high': 0.1, 'value': 0.016, 'label': r'$\rho_{\textup{DM}}$', 'lab': 'rhoDM'}
sigmaDD_init = {'low': 0., 'high': 30., 'value': 7., 'label': r'$\sigma_{\textup{DD}}$', 'lab': 'sigmaDD'}
hDD_init = {'low': 1., 'high': 1000., 'value': 30., 'label': r'$h_{\textup{DD}}$', 'lab': 'hDD'}
log_nu0_init = {'low': -1.5, 'high': 1.5, 'value': 0., 'label': r'$\log \nu_0$', 'lab': 'log_nu0'}
R_init = {'mean': 3.4E-3, 'sigma': 0.6E-3, 'value': 3.4E-3, 'label': r'$R$', 'lab': 'R'}
zsun_init = {'low': -150, 'high': 150, 'value': 0., 'label': r'$z_{\odot}$', 'lab': 'zsun'}
# init kinematic
w0_init = {'low': -15, 'high': 0., 'value': -7., 'label': r'$w_0$', 'lab': 'w0'}
log_sigmaw_init = {'low': 0., 'high': 5., 'value': 2.4, 'label': r'$\log \sigma_w$', 'lab': 'log_sigmaw'}
q_sigmaw_init = {'low': 0., 'high': 1., 'value': 0.5, 'label': r'$q_{\sigma,w}$', 'lab': 'q_sigmaw'}
log_a_init = {'low': -1., 'high': 1., 'value': np.log(0.7), 'label': r'$\log a$', 'lab': 'log_a'}
q_a_init = {'low': 0., 'high': 1., 'value': 0.5, 'label': r'$q_a$', 'lab': 'q_a'}
log_phi_init = {'low': 0., 'high': 10., 'value': 2., 'label': r'$\log \Phi$', 'lab': 'log_phi'}

init_kinematic = [w0_init, log_sigmaw_init, q_sigmaw_init, log_a_init, q_a_init, log_phi_init]
init_DM = [rhob_init, sigmaz_init, rhoDM_init, log_nu0_init, R_init, zsun_init]
init_DDDM = [rhob_init, sigmaz_init, rhoDM_init, sigmaDD_init, hDD_init, log_nu0_init, R_init, zsun_init]
init_no = [rhob_init, sigmaz_init, log_nu0_init, R_init, zsun_init]

init_dict = {
    "kin": init_kinematic,
    "DM": init_DM,
    "DDDM": init_DDDM,
    "no": init_no,
}

def flatten_array(arr):
    flattened = []
    for item in arr:
        if isinstance(item, np.ndarray):
            flattened.extend(flatten_array(item))
        else:
            flattened.append(item)
    return np.array(flattened)

def generate_init(ini):
    init = init_dict[ini]
    # print(init)
    theta = np.array([])
    locs = np.array([])
    scales = np.array([])
    labels = np.array([])
    labs = np.array([])
    for init_i in init:
        if 'mean_arr' in init_i.keys():
            # locs = np.append(locs, init_i['mean_arr'])
            # scales = np.append(scales, init_i['sigma_arr'])
            theta = np.append(theta, init_i['mean_arr'])
            if 'label' in init_i.keys():
                labels = np.append(labels, init_i['label'])
                labs = np.append(labs, init_i['lab'])
            continue
        elif 'low' in init_i.keys():
            locs = np.append(locs, init_i['low'])
            scales = np.append(scales, init_i['high'] - init_i['low'])
            theta = np.append(theta, init_i['value'])
        elif 'mean' in init_i.keys():
            locs = np.append(locs, init_i['mean'])
            scales = np.append(scales, init_i['sigma'])
            theta = np.append(theta, init_i['value'])
        else:
            raise ValueError("malformed init")
        labels = np.append(labels, init_i['label'])
        labs = np.append(labs, init_i['lab'])
    return theta, flatten_array(locs), flatten_array(scales), labels, labs

def get_data(zpath, wpath, index):
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


def get_initial_position_normal(labs, ini, chain=None, indexes=None):
    init = init_dict[ini]
    locs = {i['lab']: i['low'] for i in init}
    scales = {i['lab']: i['high'] - i['low'] for i in init}
    if chain is not None:
        for k, i in zip(labs, indexes):
            v = chain[:, :, i]
            flat_raw = v.reshape(-1)
            mad_v = mad(flat_raw)
            median_v = np.median(flat_raw)                
            if k in ["q_sigmaw", "q_a"]:
                delta_up = (1-median_v)/5 if median_v+4*mad_v > 1. else mad_v
                delta_down = (median_v)/5 if median_v-4*mad_v < 0. else mad_v
                mid = (median_v+delta_up+median_v-delta_down)/2
                delta = (delta_up+delta_down)/2
                locs[k] = mid
                scales[k] = delta
                continue
            locs[k] = median_v
            scales[k] = mad_v
    keys = np.array(list(locs.keys()))
    locs = np.array(list(locs.values()))
    scales = np.array(list(scales.values()))
    return keys, locs, scales


def run_mcmc(func, labs, indexes, data, output_path, steps0=500, burn0=300, steps=1000, burn=300, thin=20):
    # get data
    zdata, wdata = data
    # initial position
    print("Generating initial position")
    keys, locs, scales = get_initial_position_normal(labs, indexes=indexes)
    ndim = len(locs_raw)+24
    nwalkers = 10*ndim

    p0 = func.generate_p0(nwalkers, locs, scales)
    # run mcmc
    print(f"Run mcmc for preliminary, {steps0} steps each")
    time0 = time()
    for i in range(5):
        print("\t", i)
        t0 = time()
        chain = func.mcmc(steps0, nwalkers, p0, zdata, wdata,
                          locs, scales, dz=1, verbose=True, parallel=True)
        print(time() - t0, "s")
        keys, locs_normal, scales_normal = get_initial_position_normal(
            keys, chain[burn0:], np.arange(24, 24+len(keys)))
        p0 = func.generate_p0(nwalkers, locs_normal, scales_normal, norm=True)
        t0 = time()
        chain = func.mcmc(steps0, nwalkers, p0, zdata, wdata,
                          locs, scales, dz=1, verbose=True, parallel=True)
        print(time() - t0, "s")
        chain.shape
        p0 = chain[-1]
    print("total", time() - time0, "s")
    # long mcmc
    print(f"Running long MCMC, {steps} steps")
    p0 = chain[-1]
    t0 = time()
    chain = func.mcmc(steps, nwalkers, p0, zdata, wdata, locs,
                      scales, dz=1, verbose=True, parallel=True)
    print(time() - t0, "s")
    np.save(output_path, chain[burn::thin])


def run_calculate_bic_aic(func, labs, data, index, chain, output_file):
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
        f.write(f"{index},{bic},{aic},{datetime.now()}\n")



