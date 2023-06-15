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
from hammer import Model as MCMC_Model
from typing import Tuple
 
class Model(Enum):
    DM = "DM"
    DDDM = "DDDM"
    NO = "NO"

func_dict = {
    "DM": MCMC_Model.DM,
    "DDDM": MCMC_Model.DDDM,
    "NO": MCMC_Model.NO,
}

style()

current = pathlib.Path(__file__).parent.resolve()
root_dir = join(current, "..")
root_data_dir = join(root_dir, 'Data')
baryon_dir = join(root_data_dir, "Baryon")
df_baryon = vaex.open(join(baryon_dir, "baryon.hdf5"))

# init dynamics
rhob = df_baryon["rho"].to_numpy()  # Msun/pc^3
sigmaz = df_baryon["sigma_z"].to_numpy()  # km/s
rhob_e = df_baryon["e_rho"].to_numpy()  # Msun/pc^3
sigmaz_e = df_baryon["e_sigma_z"].to_numpy()  # km/s
rhob_init = {'mean_arr': rhob, 'sigma_arr': rhob_e,
             'label': r'$\rho_{b}$', 'lab': 'rhob'}
sigmaz_init = {'mean_arr': sigmaz, 'sigma_arr': sigmaz_e}
rhoDM_init = {'low': -0.05, 'high': 0.1, 'value': 0.016,
              'label': r'$\rho_{\textup{DM}}$', 'lab': 'rhoDM'}
sigmaDD_init = {'low': 0., 'high': 30., 'value': 7.,
                'label': r'$\sigma_{\textup{DD}}$', 'lab': 'sigmaDD'}
hDD_init = {'low': 1., 'high': 1000., 'value': 30.,
            'label': r'$h_{\textup{DD}}$', 'lab': 'hDD'}
log_nu0_init = {'low': -1.5, 'high': 1.5, 'value': 0.,
                'label': r'$\log \nu_0$', 'lab': 'log_nu0'}
R_init = {'mean': 3.4E-3, 'sigma': 0.6E-3,
          'value': 3.4E-3, 'label': r'$R$', 'lab': 'R'}
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
q_a_init = {'low': 0., 'high': 1., 'value': 0.5,
               'label': r'$q_a$', 'lab': 'q_a'}
log_phi_b_init = {'low': 0., 'high': 10., 'value': 2.,
                'label': r'$\log \Phi_b$', 'lab': 'log_phi_b'}

init_DM = [rhob_init, sigmaz_init, rhoDM_init, log_nu0_init, R_init, zsun_init, w0_init, log_sigmaw_init, q_sigmaw_init, log_a_init, q_a_init, log_phi_b_init]
init_DDDM = [rhob_init, sigmaz_init, rhoDM_init,
             sigmaDD_init, hDD_init, log_nu0_init, R_init, zsun_init, w0_init, log_sigmaw_init, q_sigmaw_init, log_a_init, q_a_init, log_phi_b_init]
init_no = [rhob_init, sigmaz_init, log_nu0_init, R_init, zsun_init, w0_init, log_sigmaw_init, q_sigmaw_init, log_a_init, q_a_init, log_phi_b_init]

init_dict = {
    "DM": init_DM,
    "DDDM": init_DDDM,
    "NO": init_no,
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
        if 'mean_arr' in init_i.keys():
            theta = np.append(theta, init_i['mean_arr'])
            locs = np.append(locs, init_i['mean_arr'])
            scales = np.append(scales, init_i['sigma_arr'])
            if 'label' in init_i.keys():
                labels = np.append(labels, init_i['label'])
                labs = np.append(labs, init_i['lab'])
                indexes.append(range(0, 12))
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
        indexes.append(i+22)
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


# def eprint(*args, **kwargs):
#     print(*args, file=sys.stderr, **kwargs)

def mcmc(model: Model, z_path: str, w_path: str, **options):
    """
    Run MCMC

    Parameters
    ----------
    model: `Model` = `Model.DM`, `Model.DDDM`, or `Model.NO` \n
    z_path: `str` = path to z data \n
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
    func = func_dict[model.value]
    if verbose: print("running...")
    name = z_path.split("/")[-1].replace(".hdf5", "").replace("z_", "")
    zdata = get_data(z_path)
    wdata = get_data(w_path)
    
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
        chain = func.mcmc(step0, zdata, wdata, p0, locs, scales, dz=1., parallel=True, verbose=verbose)
        t1 = time()
        if verbose: print(f"{i}: first half mcmc done {np.round(t1-t0, 2)} s")
        locs_normal, scales_normal = get_initial_position_normal(model, chain=chain[int(step0/2):])
        p0 = func.generate_p0(nwalker, locs_normal, scales_normal, norm=True)
        t0 = time()
        chain = func.mcmc(step0, zdata, wdata, p0, locs, scales, dz=1., parallel=True, verbose=verbose)
        t1 = time()
        if verbose: print(f"{i}: second half mcmc done {np.round(t1-t0, 2)} s")
        p0 = chain[-1]
    chain = func.mcmc(burn, zdata, wdata, p0, locs, scales, dz=1., parallel=True, verbose=verbose)
    p0 = chain[-1]
    chain = func.mcmc(step, zdata, wdata, p0, locs, scales, dz=1., parallel=True, verbose=verbose)
    chain_thin = chain[::thin]
    return {
        "indexes": indexes,
        "labs": labs,
        "labels": labels,
        "chain": chain_thin
        }

def calculate_prob(model: Model, zdata: Tuple[np.ndarray, np.ndarray, np.ndarray], wdata: Tuple[np.ndarray, np.ndarray, np.ndarray], flat_chain: np.ndarray, name: str, path: str, **options: dict):
    """
    Calculate maximum likelihood, BIC, and AIC, then save the result to a file

    Parameters
    ----------
    model: `Model` = `Model.DM`, `Model.DDDM`, or `Model.NO` \n
    zdata: ndarray = (zmid, znum, zerr) \n
    wdata: ndarray = (wmid, wnum, werr) \n
    flat_chain: ndarray(shape=(nsample, ndim)) \n
    path: `str` = path to save the result \n
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
    length = len(flat_chain)
    sample = flat_chain[np.random.choice(length, nsample, replace=True)]
    ndim = flat_chain.shape[1]
    # calculate likelihood
    if verbose: print("Calculating likelihood")
    probs = func.log_prob_par(sample, zdata, wdata, locs, scales, dz=1., batch=batch)
    likelihood = probs[:, 1]
    # remove nan from likelihood
    likelihood = likelihood[~np.isnan(likelihood)]
    max_likelihood = np.max(likelihood)
    # calculate BIC
    zmid = zdata[0]
    wmid = wdata[0]
    bic = -2 * max_likelihood + ndim * np.log(3*len(zmid) + 3*len(wmid))
    aic = -2 * max_likelihood + 2 * ndim
    print(f"max log-likelihood: {max_likelihood}")
    print(f"BIC: {bic}")
    print(f"AIC: {aic}")
    with open(path, 'a') as f:
        f.write(f"{name},{max_likelihood},{bic},{aic},{datetime.now()}\n")