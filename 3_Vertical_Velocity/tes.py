from glob import glob
from os.path import join, abspath
import sys
from os import pardir
from tqdm import tqdm
import vaex
import numpy as np

root_dir = abspath(pardir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from utils import style

style()

root_data_dir = join(root_dir, "Data")
vel_dir = join(root_data_dir, "Velocity-Distribution-metal")
files = glob(join(vel_dir, "w*"))
files.sort()

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

    data = (mid, num)
    return data

index = 0
file = files[index]
wdata = get_data(file)

# init kinematic
w0_init = {'low': -15, 'high': 0., 'value': -
           7., 'label': r'$w_0$', 'lab': 'w0'}
log_sigmaw_init = {'low': np.log(3), 'high': np.log(50), 'value': np.log(
    5), 'label': r'$\log \sigma_{w}$', 'lab': 'log_sigmaw'}
q_sigmaw_init = {'low': 0, 'high': 1, 'value': 0.5, 'label': r'$q_{w}$', 'lab': 'q_sigmaw'}
log_a_init = {'mean': 0., 'sigma': 2., 'value': 0.,
               'label': r'$\log a$', 'lab': 'log_a'}
q_a_init = {'low': 0.01, 'high': 1., 'value': 0.5,
               'label': r'$q_a$', 'lab': 'q_a'}
log_phi_b_init = {'low': 0., 'high': 10., 'value': 2.,
                'label': r'$\log \Phi_b$', 'lab': 'log_phi_b'}

init_kin = [w0_init, log_sigmaw_init, q_sigmaw_init, log_a_init, q_a_init, log_phi_b_init]

def flatten_array(arr: np.ndarray):
    flattened = []
    for item in arr:
        if isinstance(item, np.ndarray):
            flattened.extend(flatten_array(item))
        else:
            flattened.append(item)
    return np.array(flattened)


def generate_init_kin(log_max: float):
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
    init = init_kin

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
            if init_i['lab'] == 'log_a':
                locs = np.append(locs, log_max + init_i['mean'])
                scales = np.append(scales, init_i['sigma'])
                theta = np.append(theta, log_max + init_i['value'])
            else:    
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

from hammer import vel
log_max = np.log(wdata[1].max())
res = generate_init_kin(log_max)

theta = res['theta']
locs = res['locs']
scales = res['scales']
labs = res['labs']
labels = res['labels']

ndim = len(labs)
nwalker = 10*ndim
p0 = None
p0 = vel.generate_p0(nwalker, locs, scales)

from scipy.stats import median_abs_deviation as mad

def get_initial_position_normal(log_max: float, chain: np.ndarray):
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
    init = generate_init_kin(log_max)
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

indexes = list(range(ndim))

for _ in tqdm(range(1)):
    chain = vel.mcmc(5, p0, wdata, locs, scales, parallel=True, verbose=True)
    locs_normal, scales_normal = get_initial_position_normal(log_max, chain=chain)
    p0 = vel.generate_p0(nwalker, locs_normal, scales_normal, norm=True)
    print(p0.shape)
    chain = vel.mcmc(5, p0, wdata, locs, scales, parallel=True, verbose=True)
    p0 = chain[-1]