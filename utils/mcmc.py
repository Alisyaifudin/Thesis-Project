import numpy as np
from glob import glob
import vaex
from time import time
from os.path import join
from scipy.stats import median_abs_deviation as mad
from .style import style
from datetime import datetime
import hammer
from tqdm import tqdm
import sys
import pathlib
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
log_sigmaw1_init = {'low': np.log(1), 'high': np.log(30), 'value': np.log(
    5), 'label': r'$\log \sigma_{w,1}$', 'lab': 'log_sigmaw1'}
log_sigmaw2_init = {'low': np.log(1), 'high': np.log(30), 'value': np.log(
    10), 'label': r'$\log \sigma_{w,2}$', 'lab': 'log_sigmaw2'}
log_a1_init = {'mean': 0., 'sigma': 2., 'value': 0.,
               'label': r'$\log a_1$', 'lab': 'log_a1'}
log_a2_init = {'mean': 0., 'sigma': 2., 'value': 0.,
               'label': r'$\log a_2$', 'lab': 'log_a2'}
log_phi_init = {'low': 0., 'high': 10., 'value': 2.,
                'label': r'$\log \Phi$', 'lab': 'log_phi'}

init_kinematic = [w0_init, log_sigmaw1_init,
                  log_sigmaw2_init, log_a1_init, log_a2_init, log_phi_init]
init_DM = [rhob_init, sigmaz_init, rhoDM_init, log_nu0_init, R_init, zsun_init]
init_DDDM = [rhob_init, sigmaz_init, rhoDM_init,
             sigmaDD_init, hDD_init, log_nu0_init, R_init, zsun_init]
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
    if not ini in init_dict.keys():
        raise ValueError(f"ini must be {init_dict.keys()}")
    init = init_dict[ini]
    # print(init)
    theta = np.array([])
    locs = np.array([])
    scales = np.array([])
    labels = np.array([])
    labs = np.array([])
    for init_i in init:
        if 'mean_arr' in init_i.keys():
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


def get_data(dir_path, index, tipe):
    if not tipe in ["z", "w"]:
        raise ValueError("'tipe' should be 'z' or 'w'")
    files = glob(join(dir_path, f'{tipe}*.hdf5'))
    files.sort()

    file = files[index]
    data = vaex.open(file)

    mid = data['mid'].to_numpy()
    num = data['num'].to_numpy()
    err = data['err'].to_numpy()

    data = (mid, num, err)
    return data


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


def get_initial_position_normal(ini, chain=None, indexes=None):
    _, locs, scales, labels, labs = generate_init(ini)
    minus = 0
    if 'rhob' in labs:
        minus = 24
    if chain is not None:
        for lab, i in zip(labs, indexes):
            if lab == "rhob":
                continue
            v = chain[:, :, i]
            flat_raw = v.reshape(-1)
            mad_v = mad(flat_raw)
            median_v = np.median(flat_raw)
            locs[i-minus] = median_v
            scales[i-minus] = mad_v
    return locs, scales


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class mcmc:
    """required: index, z_dir_path, phi_dir_path, model
        optional:
            step0 = 1000
            step = 2000
            thin = 20
        .run(it=2) to run the program. it is the mcmc iteration.
        return:
            dict(indexes, labs, labels, chain)"""
    models = ["DM", "DDDM", "no"]
    funcs = {
        "DM": hammer.dm,
        "DDDM": hammer.dddm,
        "no": hammer.no
    }

    def __init__(self):
        self.ready = False
        self.props = {"index": False, "z_dir_path": False,
                      "phi_dir_path": False, "model": False}
        self._step0 = 1000
        self._step = 2000
        self._thin = 20
        self.func = None

    def index(self, p):
        self._index = p
        self.props['index'] = True
        return self

    def z_dir_path(self, p):
        self._z_dir_path = p
        self.props['z_dir_path'] = True
        return self

    def phi_dir_path(self, p):
        self._phi_dir_path = p
        self.props['phi_dir_path'] = True
        return self

    def model(self, p):
        if not p in self.models:
            raise ValueError("model must be 'DM', 'DDDM', or 'no'")
        self._model = p
        self.props['model'] = True
        self.func = self.funcs[p]
        return self

    def step0(self, p):
        self._step0 = p
        return self

    def step(self, p):
        self._step = p
        return self

    def thin(self, p):
        self._thin = p
        return self

    def run(self, it=2):
        if not all(self.props.values()):
            eprint("props: ", self.props)
            raise ValueError("all requirement must be set first")
        print("running...")
        files = glob(join(self._z_dir_path, "z*"))
        files.sort()
        file = files[self._index]
        name = file.split("/")[-1].replace(".hdf5", "").replace("z_", "")
        zdata = get_data(self._z_dir_path, self._index, "z")
        pred = np.load(join(self._phi_dir_path, f'pred_{name}.npy'))
        phis = np.load(join(self._phi_dir_path, f'phi_{name}.npy'))
        kin = (phis, pred)
        theta, locs, scales, labels, labs = generate_init(self._model)
        ndim = len(locs)+24
        indexes = [12] + list(range(24, 24+len(locs)))
        nwalker = 10*ndim
        print("mcmc...")
        p0 = None
        while True:
            p0 = self.func.generate_p0(nwalker, locs, scales)
            prob0 = self.func.log_prob_par(p0, zdata, kin, locs, scales)
            mask = np.isinf(prob0[:, 0])
            p0 = p0[~mask]
            if p0.shape[0] % 2 != 0:
                p0 = np.append(p0, p0[0][None, :], axis=0)
            if len(p0) > 2*ndim:
                break
        for i in tqdm(range(it), desc="mcmc"):
            t0 = time()
            chain = self.func.mcmc(
                self._step0, zdata, kin, p0, locs, scales, dz=1., parallel=True, verbose=True)
            t1 = time()
            print(f"{i}: first half mcmc done {np.round(t1-t0, 2)} s")
            locs_normal, scales_normal = get_initial_position_normal(
                self._model, chain=chain[int(self._step0/2):], indexes=indexes)
            while True:
                p0 = self.func.generate_p0(
                    nwalker, locs_normal, scales_normal, norm=True)
                prob0 = self.func.log_prob_par(p0, zdata, kin, locs, scales)
                mask = np.isinf(prob0[:, 0])
                p0 = p0[~mask]
                if p0.shape[0] % 2 != 0:
                    p0 = np.append(p0, p0[0][None, :], axis=0)
                if len(p0) > 2*ndim:
                    break
            t0 = time()
            chain = self.func.mcmc(
                self._step0, zdata, kin, p0, locs, scales, dz=1., parallel=True, verbose=True)
            t1 = time()
            print(f"{i}: second half mcmc done {np.round(t1-t0, 2)} s")
            p0 = chain[-1]
        chain = self.func.mcmc(self._step, zdata, kin, p0,
                               locs, scales, dz=1., parallel=True, verbose=True)
        chain_thin = chain[::self._thin]
        return {
            "indexes": indexes,
            "labs": labs,
            "labels": labels,
            "chain": chain_thin
        }


class calculate_prob:
    """required: index, z_dir_path, phi_dir_path, model, flat, path
        optional:
            batch = 10_000
        .run to run the program"""
    models = ["DM", "DDDM", "no"]
    funcs = {
        "DM": hammer.dm,
        "DDDM": hammer.dddm,
        "no": hammer.no
    }

    def __init__(self):
        self.props = {
            "index": False,
            "z_dir_path": False,
            "phi_dir_path": False,
            "model": False,
            "flat": False,
            "path": False
        }
        self._batch = 10_000

    def index(self, p):
        self._index = p
        self.props['index'] = True
        return self

    def z_dir_path(self, p):
        self._z_dir_path = p
        self.props['z_dir_path'] = True
        return self

    def phi_dir_path(self, p):
        self._phi_dir_path = p
        self.props['phi_dir_path'] = True
        return self

    def model(self, p):
        if not p in self.models:
            raise ValueError("model must be 'DM', 'DDDM', or 'no'")
        self._model = p
        self.props['model'] = True
        self.func = self.funcs[p]
        return self

    def flat(self, p):
        self._flat = p
        self.props['flat'] = True
        return self

    def path(self, p):
        self._path = p
        self.props['path'] = True
        return self

    def batch(self, p):
        self._batch = p
        return self

    def run(self):
        if not all(self.props.values()):
            eprint("props: ", self.props)
            raise ValueError("all requirement must be set first")
        print("running...")
        files = glob(join(self._z_dir_path, "z*"))
        files.sort()
        file = files[self._index]
        name = file.split("/")[-1].replace(".hdf5", "").replace("z_", "")
        zdata = get_data(self._z_dir_path, self._index, "z")
        print("opening pred file...")
        pred = np.load(join(self._phi_dir_path, f'pred_{name}.npy'))
        print("opening phis file...")
        phis = np.load(join(self._phi_dir_path, f'phi_{name}.npy'))
        kin = (phis, pred)
        _, locs, scales, _, _ = generate_init(self._model)
        # calculate likelihood
        ndim = self._flat.shape[1]
        print("Calculating likelihood")
        probs = self.func.log_prob_par(
            self._flat, zdata, kin, locs, scales, dz=1., batch=self._batch)
        likelihood = probs[:, 1]
        # remove nan from likelihood
        likelihood = likelihood[~np.isnan(likelihood)]
        max_likelihood = np.max(likelihood)
        # calculate BIC
        zmid = zdata[0]
        bic = -2 * max_likelihood + ndim * \
            np.log(3*len(zmid)+len(zmid)*pred.shape[1]*pred.shape[2])
        aic = -2 * max_likelihood + 2 * ndim
        print(f"max log-likelihood: {max_likelihood}")
        print(f"BIC: {bic}")
        print(f"AIC: {aic}")
        with open(self._path, 'a') as f:
            f.write(
                f"{self._index},{max_likelihood},{bic},{aic},{datetime.now()}\n")
