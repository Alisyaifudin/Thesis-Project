import numpy as np
import vaex
from time import time
from os.path import join
from scipy.stats import median_abs_deviation as mad
from .style import style
from datetime import datetime
from tqdm import tqdm
import pathlib
from enum import Enum
from hammer import Model as MCMC_Model, vel
from typing import Tuple
 
class Model(Enum):
    DM = "DM"
    DDDM = "DDDM"
    NO = "NO"
    KIN = "KIN"

func_dict = {
    "DM": MCMC_Model.DM,
    "DDDM": MCMC_Model.DDDM,
    "NO": MCMC_Model.NO,
}

style()

current = pathlib.Path(__file__).parent.resolve()
root_dir = join(current, "..")
root_data_dir = join(root_dir, 'Data')

# init dynamics
mu0_init = {'low': 0.01, 'high': 3.0, 'value': 1.0,
              'label': r'$\mu_0$', 'lab': 'mu0'}
rhoDM_init = {'low': -0.05, 'high': 0.1, 'value': 0.016,
              'label': r'$\rho_{\textup{DM}}$', 'lab': 'rhoDM'}
sigmaDD_init = {'low': 0., 'high': 30., 'value': 7.,
                'label': r'$\sigma_{\textup{DD}}$', 'lab': 'sigmaDD'}
hDD_init = {'low': 1., 'high': 1000., 'value': 30.,
            'label': r'$h_{\textup{DD}}$', 'lab': 'hDD'}
log_nu0_init = {'low': -1.5, 'high': 1.5, 'value': 0.,
                'label': r'$\log \nu_0$', 'lab': 'log_nu0'}
# R_init = {'mean': 3.4E-3, 'sigma': 0.6E-3,
#           'value': 3.4E-3, 'label': r'$R$', 'lab': 'R'}
zsun_init = {'low': -150, 'high': 150, 'value': 0.,
             'label': r'$z_{\odot}$', 'lab': 'zsun'}
# init kinematic
w0_init = {'low': -15, 'high': 0., 'value': -
           7., 'label': r'$w_0$', 'lab': 'w0'}
log_sigmaw_init = {'low': np.log(3), 'high': np.log(30), 'value': np.log(
    5), 'label': r'$\log \sigma_{w}$', 'lab': 'log_sigmaw'}
q_sigmaw_init = {'low': 0, 'high': 1, 'value': 0.5, 'label': r'$q_{w}$', 'lab': 'q_sigmaw'}
log_a_init = {'mean': 0., 'sigma': 2., 'value': 0.,
               'label': r'$\log a$', 'lab': 'log_a'}
q_a_init = {'low': 0.01, 'high': 1., 'value': 0.5,
               'label': r'$q_a$', 'lab': 'q_a'}
log_phi_b_init = {'low': 0., 'high': 10., 'value': 2.,
                'label': r'$\log \Phi_b$', 'lab': 'log_phi_b'}

init_DM = [rhoDM_init, log_nu0_init, zsun_init]
init_DDDM = [rhoDM_init, sigmaDD_init, hDD_init, log_nu0_init, zsun_init]
init_no = [mu0_init, log_nu0_init, zsun_init]
init_kin = [w0_init, log_sigmaw_init, q_sigmaw_init, log_a_init, q_a_init, log_phi_b_init]
init_dict = {
    "DM": init_DM,
    "DDDM": init_DDDM,
    "NO": init_no,
    "KIN": init_kin
}

def flatten_array(arr: np.ndarray):
    flattened = []
    for item in arr:
        if isinstance(item, np.ndarray):
            flattened.extend(flatten_array(item))
        else:
            flattened.append(item)
    return np.array(flattened)


def generate_init(model: Model):
    """
    Generate initial values for the model
    
    Parameters
    ----------
    model: `Model` = `Model.DM`, `Model.DDDM`, `Model.NO`

    Returns
    -------
    init: `dict` = {
        theta: `np.ndarray`, \n
        locs: `np.ndarray`, \n
        scales: `np.ndarray`, \n
        labels: `np.ndarray`, \n
        labs: `np.ndarray,` \n
        indexes: `np.ndarray` \n
        }
    """
    if not model.value in init_dict.keys():
        raise ValueError(f"model must be {init_dict.keys()}")
    init = init_dict[model.value]

    theta = np.array([])
    locs = np.array([])
    scales = np.array([])
    labs= np.array([])
    labels= np.array([])
    indexes = []

    for i, init_i in enumerate(init):
        if 'low' in init_i.keys():
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
        indexes.append(i)
    theta = flatten_array(theta)
    locs = flatten_array(locs)
    scales = flatten_array(scales)
    
    return dict(
        theta=theta, 
        locs=locs, 
        scales=scales, 
        labels=labels,
        labs=labs,
        indexes=indexes
    )


def get_data(path: str):
    """
    Get data from path

    Parameters
    ----------
    path: `str`
        path to data

    Returns
    -------
    data: `tuple` = (mid: `ndarray`, num`ndarray`: `ndarray`, err: `ndarray`)
    """
    data = vaex.open(path)

    mid = data['mid'].to_numpy()
    num = data['num'].to_numpy()
    err = data['err'].to_numpy()

    data = (mid, num, err)
    return data


def get_params(chain: np.ndarray, indexes: np.ndarray, labs: np.ndarray):
    """
    Get transform parameters from chain

    Parameters
    ----------
    chain: `ndarray(shape(nstep,nwalker,nparam))` \n
    indexes: `ndarray(shape(nparam))` \n
    labs: `ndarray(shape(nparam))` \n

    Returns
    -------
    params: `ndarray(shape(nstep,nwalker,nparam))`
    """
    params = []
    
    for lab, index in zip(labs, indexes):
        if lab == 'rhob':
            params.append(chain[:, :, index].sum(axis=2).T/1E-2)
        elif lab == 'rhoDM':
            params.append(chain[:, :, index].T/1E-2)
        elif lab == 'R':
            params.append(chain[:, :, index].T/1E-3)
        else:
            params.append(chain[:, :, index].T)
    params = np.array(params).T
    return params


def get_initial_position_normal(model: Model, chain: np.ndarray):
    """
    Get initial position for the next run in normalized form

    Parameters
    ----------
    model: `Model` = `Model.DM`, `Model.DDDM`, or `Model.NO`\n
    chain: `ndarray(shape(nstep,nwalker,nparam))`

    Returns
    -------
    init: `tuple` = (locs: `ndarray`, scales: `ndarray`)
    """
    init = generate_init(model)
    labs = init['labs']
    locs = init['locs']
    scales = init['scales']
    indexes = init['indexes']
    for lab, index in zip(labs, indexes):
        if lab == "rhob":
            continue
        v = chain[:, :, index]
        flat_raw = v.reshape(-1)
        mad_v = mad(flat_raw)
        median_v = np.median(flat_raw)
        locs[index] = median_v
        scales[index] = mad_v
    return locs, scales


def mcmc_w(w_path: str, **options):
    """
    Run MCMC

    Parameters
    ----------
    w_path: `str` = path to w data \n
    options:
        step0: `int` = 500 \n
        step: `int` = 2000 \n
        burn: `int` = 1000 \n
        it: `int` = 3 \n
        thin: `int` = 20 \n
        verbose: `bool` = True \n
        m: `int` = 10 (multiplier, `nwalker = m*ndim`)
        
    Returns
    -------
    result: `dict` = { \n
        indexes: `ndarray(shape(nparam))`, \n
        labs: `ndarray(shape(nparam))`, \n
        labels: `ndarray(shape(nparam))`, \n
        chain: `ndarray(shape(nstep,nwalker,nparam))` \n
    }
    """
    step0 = options.get("step0", 100)
    step = options.get("step", 2000)
    burn = options.get("burn", 1000)
    it = options.get("it", 3)
    thin = options.get("thin", 20)
    verbose = options.get("verbose", True)
    m = options.get("m", 10)
    if verbose: print("running...")
    name = w_path.split("/")[-1].replace(".hdf5", "").replace("z_", "")
    wdata = get_data(w_path)
    
    init = generate_init(Model.KIN)
    locs = init['locs']
    scales = init['scales']    
    indexes = init['indexes']
    labs = init['labs']
    labels = init['labels']

    ndim = len(locs)
    nwalker = m*ndim
    if verbose: print(f"mcmc... {name}")
    p0 = vel.generate_p0(nwalker, locs, scales)
    for i in tqdm(range(it), desc="mcmc"):
        t0 = time()
        chain = vel.mcmc(step0, p0, wdata,  locs, scales, parallel=True, verbose=verbose)
        t1 = time()
        if verbose: print(f"{i}: first half mcmc done {np.round(t1-t0, 2)} s")
        locs_normal, scales_normal = get_initial_position_normal(Model.KIN, chain=chain[int(step0/2):])
        p0 = vel.generate_p0(nwalker, locs_normal, scales_normal, norm=True)
        t0 = time()
        chain = vel.mcmc(step0, p0, wdata,  locs, scales, parallel=True, verbose=verbose)
        t1 = time()
        if verbose: print(f"{i}: second half mcmc done {np.round(t1-t0, 2)} s")
        p0 = chain[-1]
    chain = vel.mcmc(burn, p0, wdata,  locs, scales, parallel=True, verbose=verbose)
    p0 = chain[-1]
    chain = vel.mcmc(step, p0, wdata,  locs, scales, parallel=True, verbose=verbose)
    chain_thin = chain[::thin]
    return {
        "indexes": indexes,
        "labs": labs,
        "labels": labels,
        "chain": chain_thin
        }

def mcmc_z(model: Model, z_path: str, kin: np.ndarray, pot_b: np.ndarray, z_b: np.ndarray, **options):
    """
    Run MCMC

    Parameters
    ----------
    model: `Model` = `Model.DM`, `Model.DDDM`, or `Model.NO` \n
    z_path: `str` = path to z file \n
    pot_b: `np.ndarray` = shape(nz) \n
    z_b: `np.ndarray` = shape(nz) \n
    kin: `np.ndarray` = shape(4) \n
        - kin[0] = sigma_w1 \n
        - kin[1] = sigma_w2 \n
        - kin[2] = a1 \n
        - kin[3] = a2 \n

    options:
        step0: `int` = 500 \n
        step: `int` = 2000 \n
        burn: `int` = 1000 \n
        it: `int` = 3 \n
        thin: `int` = 20 \n
        verbose: `bool` = True \n
        m: `int` = 10 (multiplier, `nwalker = m*ndim`)
        
    Returns
    -------
    result: `dict` = { \n
        indexes: `ndarray(shape(nparam))`, \n
        labs: `ndarray(shape(nparam))`, \n
        labels: `ndarray(shape(nparam))`, \n
        chain: `ndarray(shape(nstep,nwalker,nparam))` \n
    }
    """
    step0 = options.get("step0", 100)
    step = options.get("step", 2000)
    burn = options.get("burn", 1000)
    it = options.get("it", 3)
    thin = options.get("thin", 20)
    verbose = options.get("verbose", True)
    m = options.get("m", 10)
    func = func_dict[model.value]
    if verbose: print("running...")
    name = z_path.split("/")[-1].replace(".hdf5", "").replace("z_", "")
    zdata = get_data(z_path)
    
    init = generate_init(model)
    locs = init['locs']
    scales = init['scales']    
    indexes = init['indexes']
    labs = init['labs']
    labels = init['labels']

    ndim = len(locs)
    nwalker = m*ndim
    if verbose: print(f"mcmc... {name}")
    p0 = func.generate_p0(nwalker, locs, scales)
    for i in tqdm(range(it), desc="mcmc"):
        t0 = time()
        chain = func.mcmc(step0, p0, kin, pot_b, z_b, zdata, locs, scales,  parallel=True, verbose=verbose)
        t1 = time()
        if verbose: print(f"{i}: first half mcmc done {np.round(t1-t0, 2)} s")
        locs_normal, scales_normal = get_initial_position_normal(model, chain=chain[int(step0/2):])
        p0 = func.generate_p0(nwalker, locs_normal, scales_normal, norm=True)
        t0 = time()
        chain = func.mcmc(step0, p0, kin, pot_b, z_b, zdata, locs, scales, parallel=True, verbose=verbose)
        t1 = time()
        if verbose: print(f"{i}: second half mcmc done {np.round(t1-t0, 2)} s")
        p0 = chain[-1]
    t0 = time()
    chain = func.mcmc(burn, p0, kin, pot_b, z_b, zdata, locs, scales, parallel=True, verbose=verbose)
    t1 = time()
    if verbose: print(f"burn done {np.round(t1-t0, 2)} s")
    p0 = chain[-1]
    t0 = time()
    chain = func.mcmc(step, p0, kin, pot_b, z_b, zdata, locs, scales, parallel=True, verbose=verbose)
    t1 = time()
    if verbose: print(f"mcmc done {np.round(t1-t0, 2)} s")
    chain_thin = chain[::thin]
    return {
        "indexes": indexes,
        "labs": labs,
        "labels": labels,
        "chain": chain_thin
        }

def mcmc_parallel_z(model: Model, z_path: str, kin: np.ndarray, pot_b: np.ndarray, z_b: np.ndarray, **options):
    """
    Run MCMC

    Parameters
    ----------
    model: `Model` = `Model.DM`, `Model.DDDM`, or `Model.NO` \n
    z_path: `str` = path to z file \n
    pot_b: `np.ndarray` = shape(nmcmc, nz) \n
    z_b: `np.ndarray` = shape(nz) \n
    kin: `np.ndarray` = shape(nmcmc, 4) \n
        - kin[:, 0] = sigma_w1 \n
        - kin[:, 1] = sigma_w2 \n
        - kin[:, 2] = a1 \n
        - kin[:, 3] = a2 \n

    options:
        step0: `int` = 500 \n
        step: `int` = 2000 \n
        burn: `int` = 1000 \n
        it: `int` = 3 \n
        thin: `int` = 20 \n
        verbose: `bool` = True \n
        m: `int` = 10 (multiplier, `nwalker = m*ndim`)
        
    Returns
    -------
    result: `dict` = { \n
        indexes: `ndarray(shape(nparam))`, \n
        labs: `ndarray(shape(nparam))`, \n
        labels: `ndarray(shape(nparam))`, \n
        chain: `ndarray(shape(nstep,nwalker,nparam))` \n
    }
    """
    step0 = options.get("step0", 100)
    step = options.get("step", 2000)
    burn = options.get("burn", 1000)
    it = options.get("it", 3)
    thin = options.get("thin", 20)
    verbose = options.get("verbose", True)
    m = options.get("m", 10)
    func = func_dict[model.value]
    if verbose: print("running...")
    name = z_path.split("/")[-1].replace(".hdf5", "").replace("z_", "")
    zdata = get_data(z_path)
    
    init = generate_init(model)
    locs = init['locs']
    scales = init['scales']    
    indexes = init['indexes']
    labs = init['labs']
    labels = init['labels']

    ndim = len(locs)
    nwalker = m*ndim
    n_mcmc = pot_b.shape[0]
    if verbose: print(f"mcmc... {name}")
    p0 = func.generate_p0(nwalker, locs, scales)
    p0 = np.tile(p0, (n_mcmc, 1, 1))
    print(p0.shape)
    for i in tqdm(range(it), desc="mcmc"):
        t0 = time()
        chain = func.mcmc_parallel(step0, p0, kin, pot_b, z_b, zdata, locs, scales,  parallel=True, verbose=verbose)
        t1 = time()
        if verbose: print(f"{i}: first half mcmc done {np.round(t1-t0, 2)} s")
        for j in range(n_mcmc):
            locs_normal, scales_normal = get_initial_position_normal(model, chain=chain[j, int(step0/2):])
            p0_j = func.generate_p0(nwalker, locs_normal, scales_normal, norm=True)
            p0[j] = p0_j
        t0 = time()
        chain = func.mcmc_parallel(step0, p0, kin, pot_b, z_b, zdata, locs, scales, parallel=True, verbose=verbose)
        t1 = time()
        if verbose: print(f"{i}: second half mcmc done {np.round(t1-t0, 2)} s")
        p0 = chain[:, -1]
    t0 = time()
    chain = func.mcmc_parallel(burn, p0, kin, pot_b, z_b, zdata, locs, scales, parallel=True, verbose=verbose)
    t1 = time()
    if verbose: print(f"burn done {np.round(t1-t0, 2)} s")
    p0 = chain[:, -1]
    t0 = time()
    chain = func.mcmc_parallel(step, p0, kin, pot_b, z_b, zdata, locs, scales, parallel=True, verbose=verbose)
    t1 = time()
    if verbose: print(f"mcmc done {np.round(t1-t0, 2)} s")
    chain_thin = chain[:, ::thin]
    return {
        "indexes": indexes,
        "labs": labs,
        "labels": labels,
        "chain": chain_thin
        }

def calculate_prob(
        model: Model, 
        flat_chains: np.ndarray, 
        zdata: Tuple[np.ndarray, np.ndarray, np.ndarray], 
        kin: np.ndarray, 
        pot_b: np.ndarray,
        z_b: np.ndarray,
        **options: dict
    ):
    """
    Calculate maximum likelihood, BIC, and AIC, then save the result to a file

    Parameters
    ----------
    model: `Model` = `Model.DM`, `Model.DDDM`, or `Model.NO` \n
    flat_chain: ndarray(shape=(n_mcmc, length, ndim)) \n
    zdata: ndarray = (zmid, znum, zerr) \n
    kin: ndarray = (n_mcmc, kin) \n
    pot_b: ndarray = (n_mcmc, pot_b) \n
    z_b: ndarray = (n_mcmc, z_b) \n
    options:
        nsample: `int` = 10_000 \n
        verbose: `bool` = True \n
        batch: `int` = 1000 \n
    """
    nsample = options.get("nsample", 10_000)
    verbose = options.get("verbose", True)
    batch = options.get("batch", 1000)
    func = func_dict[model.value]
    if verbose: print("Opening the data")
    init = generate_init(model)
    locs = init['locs']
    scales = init['scales']
    if verbose: print("Opening the chain")
    n_mcmc, length, ndim = flat_chains.shape
    # calculate likelihood
    if verbose: print("Calculating likelihood")
    mx_l = np.empty(n_mcmc)
    bics = np.empty(n_mcmc)
    aics = np.empty(n_mcmc)
    for i in tqdm(range(n_mcmc)):
        ind = np.random.choice(length, nsample, replace=True)
        theta = flat_chains[i, ind]
        probs = func.log_prob_par(theta, kin[i], zdata, pot_b[i], z_b, locs, scales, batch=batch)
        likelihood = probs[:, 1]
        # remove nan from likelihood
        likelihood = likelihood[~np.isnan(likelihood)]
        max_likelihood = np.max(likelihood)
        # calculate BIC
        zmid = zdata[0]
        bic = -2 * max_likelihood + ndim * np.log(3*len(zmid))
        aic = -2 * max_likelihood + 2 * ndim
        mx_l[i] = max_likelihood
        bics[i] = bic
        aics[i] = aic
    df = vaex.from_arrays(max_likelihood=mx_l, bic=bics, aic=aics)
    return df