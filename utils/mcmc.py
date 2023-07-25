import numpy as np
import vaex 
from time import time
from os.path import join
from scipy.stats import median_abs_deviation as mad
from .style import style
from datetime import datetime
from tqdm import tqdm
import pathlib
from hammer import Model
from typing import Tuple
# from abc import ABC, abstractmethod

# class Model(Enum):
#     DM = HammerModel.DM
#     DDDM = HammerModel.DDDM
#     NO = HammerModel.NO

# class Model(ABC):
#     @abstractmethod
#     def generate(self):
#         pass

# func_dict = {
#     Model.DM.value: HammerModel.DM,
#     Model.DDDM.value: HammerModel.DDDM,
#     Model.NO.value: HammerModel.NO,
# }

style()

current = pathlib.Path(__file__).parent.resolve()
root_dir = join(current, "..")
root_data_dir = join(root_dir, 'Data')
baryon_dir = join(root_data_dir, "Baryon")
# load baryons components
df_baryon = vaex.open(join(baryon_dir, "baryon.hdf5"))
rhob_mean = np.array(df_baryon["rho"].to_numpy())  # Msun/pc^3
sigmaz_mean = df_baryon["sigma_z"].to_numpy() # km/s
rhob_std = np.array(df_baryon["e_rho"].to_numpy())  # Msun/pc^3
sigmaz_std = df_baryon["e_sigma_z"].to_numpy() # km/s

# init dynamics
sigmaz_init = {'mean_arr': sigmaz_mean, 'sigma_arr': sigmaz_std, 'value': sigmaz_mean, 'label': r'$\sigma_{z}$', 'lab': 'sigmaz'}
rhob_init = {'mean_arr': rhob_mean, 'sigma_arr': rhob_std, 'value': rhob_mean, 'label': r'$\rho_{\textup{b}}$', 'lab': 'rhob'}
rhoDM_init = {'low': -0.05, 'high': 0.15, 'value': 0.016,
              'label': r'$\rho_{\textup{DM}}$', 'lab': 'rhoDM'}
sigmaDD_init = {'low': 0., 'high': 30., 'value': 7.,
                'label': r'$\sigma_{\textup{DD}}$', 'lab': 'sigmaDD'}
hDD_init = {'low': 1., 'high': 150., 'value': 30.,
            'label': r'$h_{\textup{DD}}$', 'lab': 'hDD'}
log_nu0_init = {'mean': 0, 'sigma': 3, 'value': 0.,
                'label': r'$\log \nu_0$', 'lab': 'log_nu0'}
R_init = {'mean': 3.4E-3, 'sigma': 0.6E-3,
          'value': 3.4E-3, 'label': r'$R$', 'lab': 'R'}
zsun_init = {'low': -150, 'high': 150, 'value': 0.,
             'label': r'$z_{\odot}$', 'lab': 'zsun'}
# init kinematic
w0_init = {'low': -15, 'high': 0., 'value': -7., 'label': r'$w_0$', 'lab': 'w0'}
log_sigmaw_init = {'low': np.log(3), 'high': np.log(50), 'value': np.log(
    5), 'label': r'$\log \sigma_{w}$', 'lab': 'log_sigmaw'}
q_sigmaw_init = {'low': 0, 'high': 1, 'value': 0.5, 'label': r'$q_{w}$', 'lab': 'q_sigmaw'}
log_a_init = {'mean': 0., 'sigma': 2., 'value': 0.,
               'label': r'$\log a$', 'lab': 'log_a'}
q_a_init = {'low': 0.5, 'high': 1., 'value': 0.75,
               'label': r'$q_a$', 'lab': 'q_a'}

init_DM = [sigmaz_init, rhob_init, rhoDM_init, log_nu0_init, zsun_init, R_init, w0_init, log_sigmaw_init, q_sigmaw_init, log_a_init, q_a_init]
init_DDDM = [sigmaz_init, rhob_init, rhoDM_init, sigmaDD_init, hDD_init, log_nu0_init, zsun_init, R_init, w0_init, log_sigmaw_init, q_sigmaw_init, log_a_init, q_a_init]
init_no = [sigmaz_init, rhob_init, log_nu0_init, zsun_init, R_init, w0_init, log_sigmaw_init, q_sigmaw_init, log_a_init, q_a_init]

init_dict = {
    Model.DM.name: init_DM,
    Model.DDDM.name: init_DDDM,
    Model.NO.name: init_no,
}

def flatten_array(arr: np.ndarray):
    flattened = []
    for item in arr:
        if isinstance(item, np.ndarray):
            flattened.extend(flatten_array(item))
        else:
            flattened.append(item)
    return np.array(flattened)


def generate_init(model: Model, log_nu0_max: float, log_a_max: float):
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
    # if not model in init_dict.keys():
    #     raise ValueError(f"model must be {init_dict.keys()}")
    init = init_dict[model.name]

    theta = np.array([])
    locs = np.array([])
    scales = np.array([])
    labs= np.array([])
    labels= np.array([])
    indexes = np.array([], dtype=int)
    i = 0
    for init_i in init:
        if 'mean_arr' in init_i.keys():
            lab = init_i['lab']
            label = init_i['label']
            for loc, scale, val in zip(init_i['mean_arr'], init_i['sigma_arr'], init_i['value']):
                locs = np.append(locs, loc)
                scales = np.append(scales, scale)
                theta = np.append(theta, val)
                labels = np.append(labels, label)
                labs = np.append(labs, lab)
                indexes = np.append(indexes, i)
                i += 1
            continue
        elif 'low' in init_i.keys():
            locs = np.append(locs, init_i['low'])
            scales = np.append(scales, init_i['high'] - init_i['low'])
            theta = np.append(theta, init_i['value'])
        elif 'mean' in init_i.keys():
            if init_i['lab'] == 'log_a':
                locs = np.append(locs, log_a_max + init_i['mean'])
                scales = np.append(scales, init_i['sigma'])
                theta = np.append(theta, log_a_max + init_i['value'])
            elif init_i['lab'] == 'log_nu0':
                locs = np.append(locs, log_nu0_max + init_i['mean'])
                scales = np.append(scales, init_i['sigma'])
                theta = np.append(theta, log_nu0_max + init_i['value'])
            else:    
                locs = np.append(locs, init_i['mean'])
                scales = np.append(scales, init_i['sigma'])
                theta = np.append(theta, init_i['value'])
        else:
            raise ValueError("malformed init")
        labels = np.append(labels, init_i['label'])
        labs = np.append(labs, init_i['lab'])
        indexes = np.append(indexes, i)
        i += 1
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


def get_data_w(path: str):
    """
    Get data from path

    Parameters
    ----------
    path: `str`
        path to data

    Returns
    -------
    data: `tuple` = (mid: `ndarray`, num`ndarray`: `ndarray`)
    """
    data = vaex.open(path)

    mid = data['mid'].to_numpy()
    num = data['num'].to_numpy()

    data = (mid, num)
    return data

def get_data_z(path: str):
    """
    Get data from path

    Parameters
    ----------
    path: `str`
        path to data

    Returns
    -------
    data: `tuple` = (mid: `ndarray`, num`ndarray`: `ndarray`)
    """
    data = vaex.open(path)

    mid = data['mid'].to_numpy()
    num = data['num'].to_numpy()
    com = data['com'].to_numpy()

    data = (mid, num, com)
    return data


def get_params(chain: np.ndarray, indexes: np.ndarray, labs: np.ndarray, labels: np.ndarray):
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
            params.append(chain[:, :, index].T/1E-2)
        elif lab == 'rhoDM':
            params.append(chain[:, :, index].T/1E-2)
        elif lab == 'R':
            params.append(chain[:, :, index].T/1E-3)
        else:
            params.append(chain[:, :, index].T)
    
    params = np.array(params).T
    rhob = params[:, :, 12:24]
    sigmaz = params[:, :, :12]
    rhob_tot = rhob.sum(axis=2)
    sigmaz_eff = np.sqrt(np.sum(sigmaz**2*rhob, axis=2)/rhob_tot)

    params_2 = np.append(sigmaz_eff[:, :, None], rhob_tot[:, :, None], axis=2)
    params = np.append(params_2, params[:, :, 24:], axis=2)
    labels = np.append([r"$\sigma_z$", r"$\rho_b$"], labels[24:])
    return params, labels


def get_initial_position_normal(model, log_nu0_max, log_a_max, chain: np.ndarray):
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
    init = generate_init(model, log_nu0_max, log_a_max)
    labs = init['labs']
    locs = init['locs']
    scales = init['scales']
    indexes = init['indexes']
    for lab, index in zip(labs, indexes):
        if lab in ["rhob", "sigmaz"]:
            continue
        v = chain[:, :, index]
        flat_raw = v.reshape(-1)
        mad_v = mad(flat_raw)
        median_v = np.median(flat_raw)
        locs[index] = median_v
        scales[index] = mad_v
    return locs, scales

def mcmc(
        model: Model, 
        zdata: Tuple[np.ndarray, np.ndarray], 
        wdata: Tuple[np.ndarray, np.ndarray],
        **options):
    """
    Run MCMC

    Parameters
    ----------
    model: `Model` = `Model.DM`, `Model.DDDM`, or `Model.NO` \n
    zdata: `Tuple[np.ndarray, np.ndarray]` = (zmid, znum) \n
    wdata: `Tuple[np.ndarray, np.ndarray]` = (wmid, wnum) \n
    baryon: `np.ndarray` = [...rhob, ...sigmaz] \n
    
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
    log_nu0_max = np.log(zdata[1].max())
    log_a_max = np.log(wdata[1].max())
    init = generate_init(model, log_nu0_max, log_a_max)
    locs = init['locs']
    scales = init['scales']    
    indexes = init['indexes']
    labs = init['labs']
    labels = init['labels']
    # raise NotImplementedError()
    ndim = len(locs)
    nwalker = m*ndim
    if verbose: print(f"mcmc...")
    p0 = model.generate_p0(nwalker*2, locs, scales)
    prob = model.log_prob_par(p0, zdata, wdata, locs, scales)
    mask = np.isinf(prob[:, 0])
    p0 = p0[~mask]
    if len(p0) % 2 != 0:
        p0 = p0[:-1]
    print(p0.shape)
    for i in tqdm(range(it), desc="mcmc"):
        t0 = time()
        chain = model.mcmc(step0, p0, zdata, wdata, locs, scales,  parallel=True, verbose=verbose)
        t1 = time()
        if verbose: print(f"{i}: first half mcmc done {np.round(t1-t0, 2)} s")
        locs_normal, scales_normal = get_initial_position_normal(model, log_nu0_max, log_a_max, chain=chain[int(step0/2):])
        p0 = model.generate_p0(nwalker, locs_normal, scales_normal, norm=True)
        t0 = time()
        chain = model.mcmc(step0, p0, zdata, wdata, locs, scales,  parallel=True, verbose=verbose)
        t1 = time()
        if verbose: print(f"{i}: second half mcmc done {np.round(t1-t0, 2)} s")
        p0 = chain[-1]
        prob = model.log_prob_par(p0, zdata, wdata, locs, scales)
        mask = np.isinf(prob[:, 0])
        p0 = p0[~mask]
        if len(p0) % 2 != 0:
            p0 = p0[:-1]
    t0 = time()
    chain = model.mcmc(burn, p0, zdata, wdata, locs, scales,  parallel=True, verbose=verbose)
    t1 = time()
    if verbose: print(f"burn done {np.round(t1-t0, 2)} s")
    p0 = chain[-1]
    t0 = time()
    chain = model.mcmc(step, p0, zdata, wdata, locs, scales,  parallel=True, verbose=verbose)
    t1 = time()
    if verbose: print(f"mcmc done {np.round(t1-t0, 2)} s")
    chain_thin = chain[::thin]
    return {
        "indexes": indexes,
        "labs": labs,
        "labels": labels,
        "chain": chain_thin
        }

def predictive_posterior(
        model: Model, 
        flat_chain: np.ndarray, 
        zdata: Tuple[np.ndarray, np.ndarray], 
        **options: dict
    ):
    """
    Calculate the predictive posterior of new data

    Parameters
    ----------
    model: `Model` = `Model.DM`, `Model.DDDM`, or `Model.NO` \n
    flat_chain: ndarray(shape=(length, ndim)) \n
    zdata: ndarray = (zmid, znum, comp) \n
    wdata: ndarray = (wmid, wnum) \n
    options:
        nsample: `int` = 10_000 \n
        verbose: `bool` = True \n
        batch: `int` = 1000 \n
    """
    nsample = options.get("nsample", 20_000)
    verbose = options.get("verbose", True)
    batch = options.get("batch", 1000)
    length, _ = flat_chain.shape
    if verbose: print("Calculating...")
    inds = np.random.choice(np.arange(length), size=nsample, replace=False)
    theta = flat_chain[inds]
    prob = model.predictive_posterior(theta, zdata, batch=batch)
    return prob

def bayes_factor(
        model: Model, 
        flat_chain: np.ndarray, 
        zdata: Tuple[np.ndarray, np.ndarray, np.ndarray],
        wdata:Tuple[np.ndarray, np.ndarray],
        **options: dict
    ):
    """
    Calculate the bayes factor of the model
    """
    nsample = options.get("nsample", 5_000)
    alpha = options.get("alpha", 5)
    batch = options.get("batch", 10)
    run = options.get("run", 10)
    
    length, ndim = flat_chain.shape

    dm_label = ['dm', 'log_nu0', 'zsun', 'R', 'w0', 'log_sigmaw', 'q_sigmaw', 'log_a', 'q_a']
    dddm_label = ['dm', 'sigmaDD', 'hDD', 'log_nu0', 'zsun', 'R', 'w0', 'log_sigmaw', 'q_sigmaw', 'log_a', 'q_a']
    no_label = ['log_nu0', 'zsun', 'R', 'w0', 'log_sigmaw', 'q_sigmaw', 'log_a', 'q_a']
    labels = dm_label
    if ndim == 35:
        labels = dddm_label
    elif ndim == 32:
        labels = no_label

    res = []
    for i in tqdm(range(run)):
        ind = np.random.choice(np.arange(length), size=nsample)
        theta = flat_chain[ind]
        vol = 1
        for i in range(24):
            l, u = np.percentile(theta[:, i], [alpha/2, 100-alpha/2])
            mask = (theta[:, i] > l)*(theta[:, i] < u)
            theta = theta[mask]
            vol *= u-l
        for i, label in enumerate(labels):
            if label in ['dm', 'log_nu0', 'zsun', 'R', 'w0', 'sigmaDD', 'hDD']:
                l, u = np.percentile(theta[:, i+24], [alpha/2, 100-alpha/2])
                mask = (theta[:, i+24] > l)*(theta[:, i+24] < u)
                theta = theta[mask]
                vol *= u-l
            if label == 'log_sigmaw':
                log_sigmaw = theta[:, i+24]
                q_sigmaw = theta[:, i+25]
                log_sigmaw2 = log_sigmaw - np.log(q_sigmaw)
                log_sigmaw_low, log_sigmaw_up = np.percentile(log_sigmaw, [alpha/2, 100-alpha/2])
                log_sigmaw2_low, log_sigmaw2_up = np.percentile(log_sigmaw2, [alpha/2, 100-alpha/2])
                mask = (log_sigmaw > log_sigmaw_low)*(log_sigmaw < log_sigmaw_up)*(log_sigmaw2 > log_sigmaw2_low)*(log_sigmaw2 < log_sigmaw2_up)
                theta = theta[mask]
                vol *= log_sigmaw_up-log_sigmaw_low
                vol *= log_sigmaw2_up-log_sigmaw2_low
            if label == 'log_a':
                log_a = theta[:, i+24]
                q_a = theta[:, i+25]
                log_a2 = log_a + np.log(1-q_a)
                log_a_low, log_a_up = np.percentile(log_a, [alpha/2, 100-alpha/2])
                log_a2_low, log_a2_up = np.percentile(log_a2, [alpha/2, 100-alpha/2])
                mask = (log_a > log_a_low)*(log_a < log_a_up)*(log_a2 > log_a2_low)*(log_a2 < log_a2_up)
                theta = theta[mask]
                vol *= log_a_up-log_a_low
                vol *= log_a2_up-log_a2_low
        # print(theta.shape)
        log_nu0_max = np.log(zdata[1].max())
        log_a_max = np.log(wdata[1].max())
        init = generate_init(model, log_nu0_max, log_a_max)
        locs = init['locs']
        scales = init['scales']    
        log_prob = -1*model.log_prob_par(theta, zdata, wdata, locs, scales, batch=batch)
        log_post = log_prob[:, 2]
        log_max = np.max(log_post)
        log_post -= log_max
        log_sum = log_max + np.log(np.sum(np.exp(log_post)))
        logZ = np.log(vol) - log_sum
        res.append(logZ*np.log10(np.e))
    return np.mean(res), np.std(res)

def bic_aic(
        model: Model, 
        flat_chain: np.ndarray, 
        zdata,
        wdata,
        **options: dict
    ):
    """
    Calculate the bayes factor of the model
    """
    batch = options.get("batch", 10)
    
    length, ndim = flat_chain.shape
    theta = flat_chain

    zmid, znum, comp = zdata
    wmid, wnum = wdata
    log_nu0_max = np.log(znum.max())
    log_a_max = np.log(wnum.max())
    init = generate_init(model, log_nu0_max, log_a_max)
    locs = init['locs']
    scales = init['scales']    
    log_prob = np.log10(np.e)*model.log_prob_par(theta, zdata, wdata, locs, scales, batch=batch)
    log_likelihood = log_prob[:, 1]
    log_max = np.max(log_likelihood)
    bic = -2*log_max + ndim*np.log10(len(zmid)*3)
    aic = -2*log_max + 2*ndim
    return bic, aic

# def mcmc_z(model: Model, z_path: str, psi: np.ndarray, **options):
#     """
#     Run MCMC

#     Parameters
#     ----------
#     model: `Model` = `Model.DM`, `Model.DDDM`, or `Model.NO` \n
#     z_path: `str` = path to z file \n
#     psi: `np.ndarray` = shape(4) \n

#     options:
#         step0: `int` = 500 \n
#         step: `int` = 2000 \n
#         burn: `int` = 1000 \n
#         it: `int` = 3 \n
#         thin: `int` = 20 \n
#         verbose: `bool` = True \n
#         m: `int` = 10 (multiplier, `nwalker = m*ndim`)
        
#     Returns
#     -------
#     result: `dict` = { \n
#         indexes: `ndarray(shape(nparam))`, \n
#         labs: `ndarray(shape(nparam))`, \n
#         labels: `ndarray(shape(nparam))`, \n
#         chain: `ndarray(shape(nstep,nwalker,nparam))` \n
#     }
#     """
#     step0 = options.get("step0", 100)
#     step = options.get("step", 2000)
#     burn = options.get("burn", 1000)
#     it = options.get("it", 3)
#     thin = options.get("thin", 20)
#     verbose = options.get("verbose", True)
#     m = options.get("m", 10)
#     func = func_dict[model.value]
#     if verbose: print("running...")
#     name = z_path.split("/")[-1].replace(".hdf5", "").replace("z_", "")
#     zdata = get_data(z_path)
    
#     init = generate_init(model)
#     locs = init['locs']
#     scales = init['scales']    
#     indexes = init['indexes']
#     labs = init['labs']
#     labels = init['labels']

#     ndim = len(locs)
#     nwalker = m*ndim
#     if verbose: print(f"mcmc... {name}")
#     p0 = func.generate_p0(nwalker, locs, scales)
#     for i in tqdm(range(it), desc="mcmc"):
#         t0 = time()
#         chain = func.mcmc(step0, p0, psi, zdata, locs, scales,  parallel=True, verbose=verbose)
#         t1 = time()
#         if verbose: print(f"{i}: first half mcmc done {np.round(t1-t0, 2)} s")
#         locs_normal, scales_normal = get_initial_position_normal(model, chain=chain[int(step0/2):])
#         p0 = func.generate_p0(nwalker, locs_normal, scales_normal, norm=True)
#         t0 = time()
#         chain = func.mcmc(step0, p0, psi, zdata, locs, scales,  parallel=True, verbose=verbose)
#         t1 = time()
#         if verbose: print(f"{i}: second half mcmc done {np.round(t1-t0, 2)} s")
#         p0 = chain[-1]
#     t0 = time()
#     chain = func.mcmc(burn, p0, psi, zdata, locs, scales,  parallel=True, verbose=verbose)
#     t1 = time()
#     if verbose: print(f"burn done {np.round(t1-t0, 2)} s")
#     p0 = chain[-1]
#     t0 = time()
#     chain = func.mcmc(step, p0, psi, zdata, locs, scales,  parallel=True, verbose=verbose)
#     t1 = time()
#     if verbose: print(f"mcmc done {np.round(t1-t0, 2)} s")
#     chain_thin = chain[::thin]
#     return {
#         "indexes": indexes,
#         "labs": labs,
#         "labels": labels,
#         "chain": chain_thin
#         }

# def mcmc_parallel_z(model: Model, z_path: str, psi: np.ndarray, **options):
#     """
#     Run MCMC

#     Parameters
#     ----------
#     model: `Model` = `Model.DM`, `Model.DDDM`, or `Model.NO` \n
#     z_path: `str` = path to z file \n
#     psi: `np.ndarray` = shape(nmcmc, 4) \n

#     options:
#         step0: `int` = 500 \n
#         step: `int` = 2000 \n
#         burn: `int` = 1000 \n
#         it: `int` = 3 \n
#         thin: `int` = 20 \n
#         verbose: `bool` = True \n
#         m: `int` = 10 (multiplier, `nwalker = m*ndim`)
        
#     Returns
#     -------
#     result: `dict` = { \n
#         indexes: `ndarray(shape(nparam))`, \n
#         labs: `ndarray(shape(nparam))`, \n
#         labels: `ndarray(shape(nparam))`, \n
#         chain: `ndarray(shape(nstep,nwalker,nparam))` \n
#     }
#     """
#     step0 = options.get("step0", 100)
#     step = options.get("step", 2000)
#     burn = options.get("burn", 1000)
#     it = options.get("it", 3)
#     thin = options.get("thin", 20)
#     verbose = options.get("verbose", True)
#     m = options.get("m", 10)
#     func = func_dict[model.value]
#     if verbose: print("running...")
#     name = z_path.split("/")[-1].replace(".hdf5", "").replace("z_", "")
#     zdata = get_data(z_path)
    
#     init = generate_init(model)
#     locs = init['locs']
#     scales = init['scales']    
#     indexes = init['indexes']
#     labs = init['labs']
#     labels = init['labels']

#     ndim = len(locs)
#     nwalker = m*ndim
#     n_mcmc = psi.shape[0]
#     if verbose: print(f"mcmc... {name}")
#     p0 = func.generate_p0(nwalker, locs, scales)
#     p0 = np.tile(p0, (n_mcmc, 1, 1))
#     print(p0.shape)
#     for i in tqdm(range(it), desc="mcmc"):
#         t0 = time()
#         chain = func.mcmc_parallel(step0, p0, psi, zdata, locs, scales,  parallel=True, verbose=verbose)
#         t1 = time()
#         if verbose: print(f"{i}: first half mcmc done {np.round(t1-t0, 2)} s")
#         for j in range(n_mcmc):
#             locs_normal, scales_normal = get_initial_position_normal(model, chain=chain[j, int(step0/2):])
#             p0_j = func.generate_p0(nwalker, locs_normal, scales_normal, norm=True)
#             p0[j] = p0_j
#         t0 = time()
#         chain = func.mcmc_parallel(step0, p0, psi, zdata, locs, scales, parallel=True, verbose=verbose)
#         t1 = time()
#         if verbose: print(f"{i}: second half mcmc done {np.round(t1-t0, 2)} s")
#         p0 = chain[:, -1]
#     t0 = time()
#     chain = func.mcmc_parallel(burn, p0, psi, zdata, locs, scales, parallel=True, verbose=verbose)
#     t1 = time()
#     if verbose: print(f"burn done {np.round(t1-t0, 2)} s")
#     p0 = chain[:, -1]
#     t0 = time()
#     chain = func.mcmc_parallel(step, p0, psi, zdata, locs, scales, parallel=True, verbose=verbose)
#     t1 = time()
#     if verbose: print(f"mcmc done {np.round(t1-t0, 2)} s")
#     chain_thin = chain[:, ::thin]
#     return {
#         "indexes": indexes,
#         "labs": labs,
#         "labels": labels,
#         "chain": chain_thin
#         }

# def mcmc_w(w_path: str, **options):
#     """
#     Run MCMC

#     Parameters
#     ----------
#     w_path: `str` = path to w data \n
#     options:
#         step0: `int` = 500 \n
#         step: `int` = 2000 \n
#         burn: `int` = 1000 \n
#         it: `int` = 3 \n
#         thin: `int` = 20 \n
#         verbose: `bool` = True \n
#         m: `int` = 10 (multiplier, `nwalker = m*ndim`)
        
#     Returns
#     -------
#     result: `dict` = { \n
#         indexes: `ndarray(shape(nparam))`, \n
#         labs: `ndarray(shape(nparam))`, \n
#         labels: `ndarray(shape(nparam))`, \n
#         chain: `ndarray(shape(nstep,nwalker,nparam))` \n
#     }
#     """
#     step0 = options.get("step0", 100)
#     step = options.get("step", 2000)
#     burn = options.get("burn", 1000)
#     it = options.get("it", 3)
#     thin = options.get("thin", 20)
#     verbose = options.get("verbose", True)
#     m = options.get("m", 10)
#     if verbose: print("running...")
#     name = w_path.split("/")[-1].replace(".hdf5", "").replace("z_", "")
#     wdata = get_data(w_path)
    
#     init = generate_init(Model.KIN)
#     locs = init['locs']
#     scales = init['scales']    
#     indexes = init['indexes']
#     labs = init['labs']
#     labels = init['labels']

#     ndim = len(locs)
#     nwalker = m*ndim
#     if verbose: print(f"mcmc... {name}")
#     p0 = vel.generate_p0(nwalker, locs, scales)
#     for i in tqdm(range(it), desc="mcmc"):
#         t0 = time()
#         chain = vel.mcmc(step0, p0, wdata,  locs, scales, parallel=True, verbose=verbose)
#         t1 = time()
#         if verbose: print(f"{i}: first half mcmc done {np.round(t1-t0, 2)} s")
#         locs_normal, scales_normal = get_initial_position_normal(Model.KIN, chain=chain[int(step0/2):])
#         p0 = vel.generate_p0(nwalker, locs_normal, scales_normal, norm=True)
#         t0 = time()
#         chain = vel.mcmc(step0, p0, wdata,  locs, scales, parallel=True, verbose=verbose)
#         t1 = time()
#         if verbose: print(f"{i}: second half mcmc done {np.round(t1-t0, 2)} s")
#         p0 = chain[-1]
#     chain = vel.mcmc(burn, p0, wdata,  locs, scales, parallel=True, verbose=verbose)
#     p0 = chain[-1]
#     chain = vel.mcmc(step, p0, wdata,  locs, scales, parallel=True, verbose=verbose)
#     chain_thin = chain[::thin]
#     return {
#         "indexes": indexes,
#         "labs": labs,
#         "labels": labels,
#         "chain": chain_thin
#         }
 

# def calculate_prob(
#         model: Model, 
#         flat_chains: np.ndarray, 
#         zdata: Tuple[np.ndarray, np.ndarray, np.ndarray], 
#         psi: np.ndarray, 
#         **options: dict
#     ):
#     """
#     Calculate maximum likelihood, BIC, and AIC, then save the result to a file

#     Parameters
#     ----------
#     model: `Model` = `Model.DM`, `Model.DDDM`, or `Model.NO` \n
#     flat_chain: ndarray(shape=(n_mcmc, length, ndim)) \n
#     zdata: ndarray = (zmid, znum, zerr) \n
#     psi: ndarray = (n_mcmc, kin) \n
#     options:
#         nsample: `int` = 10_000 \n
#         verbose: `bool` = True \n
#         batch: `int` = 1000 \n
#     """
#     nsample = options.get("nsample", 10_000)
#     verbose = options.get("verbose", True)
#     batch = options.get("batch", 1000)
#     func = func_dict[model.value]
#     if verbose: print("Opening the data")
#     init = generate_init(model)
#     locs = init['locs']
#     scales = init['scales']
#     if verbose: print("Opening the chain")
#     n_mcmc, length, ndim = flat_chains.shape
#     print(flat_chains.shape)
#     # flat_chain = chain_thin.reshape(n_mcmc, -1, ndim)
#     # calculate likelihood
#     if verbose: print("Calculating likelihood")
#     probs = func.log_prob_par(flat_chains, psi, zdata, locs, scales, batch=batch)
#     mx_l = np.empty(n_mcmc)
#     bics = np.empty(n_mcmc)
#     aics = np.empty(n_mcmc)
#     for i, prob in enumerate(tqdm(probs)):
#         likelihood = prob[1]
#         # remove nan from likelihood
#         likelihood = likelihood[~np.isnan(likelihood)]
#         max_likelihood = np.max(likelihood)
#         # calculate BIC
#         zmid = zdata[0]
#         bic = -2 * max_likelihood + ndim * np.log(3*len(zmid))
#         aic = -2 * max_likelihood + 2 * ndim
#         mx_l[i] = max_likelihood
#         bics[i] = bic
#         aics[i] = aic
#     df = vaex.from_arrays(max_likelihood=mx_l, bic=bics, aic=aics)
#     return df