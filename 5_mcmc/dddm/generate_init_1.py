import warnings
warnings.filterwarnings('ignore')
import numpy as np
from os.path import abspath, join
import sys
from hammer import dddm
root_dir = abspath(join('..', '..'))
data_dir = join(root_dir, 'Data', 'MCMC', 'dddm')
sys.path.append(root_dir)
from init import init

model = 1

ndim, nwalkers = init(model)

locs = dict(
    rhoDM=-0.05,
    sigmaDD=1,
    log_hDD=0,
    log_nu0=4,
    R=3.4E-3,
    zsun=-50,
    w0=-10,
    log_sigmaw=0,
    log_a=-2
)

scales = dict(
    rhoDM=0.15,
    sigmaDD=9,
    log_hDD=np.log(100),
    log_nu0=2,
    R=0.6E-3,
    zsun=100,
    w0=5,
    log_sigmaw=3,
    log_a=2
)

keys = list(locs.keys())
locs = np.array(list(locs.values()))
scales = np.array(list(scales.values()))

check = [(k, loc, s+loc) for k, loc, s in zip(keys, locs, scales)]
print(check)

print('generating p0')
p0 = dddm.generate_p0(nwalkers, locs, scales, kind=model)

np.save(join(data_dir, 'data', f'p0-{model}.npy'), p0)
np.save(join(data_dir, 'data', f'locs-{model}.npy'), locs)
np.save(join(data_dir, 'data', f'scales-{model}.npy'), scales)