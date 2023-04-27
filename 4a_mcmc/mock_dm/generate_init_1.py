import warnings
warnings.filterwarnings('ignore')
import numpy as np
from os.path import abspath, join
import sys
from hammer import dm
root_dir = abspath(join('..', '..')) 
data_dir = join(root_dir, 'Data', 'MCMC', 'dm_mock')
from init import init

model = 1
tipe = sys.argv[1]
if not tipe in ['z', 'n']:
    raise ValueError('tipe must be "z" or "n"')

ndim, nwalkers = init(model)

locs = dict(
    rhoDM=-0.05,
    log_nu0=0.5,
    R=3.4E-3,
    zsun=-10,
    w0=-9,
    log_sigmaw=1.5,
    log_a=-1.5
)

scales = dict(
    rhoDM=0.15,
    log_nu0=2,
    R=0.6E-3,
    zsun=60,
    w0=5,
    log_sigmaw=0.5,
    log_a=1.3
)

keys = list(locs.keys())
locs = np.array(list(locs.values()))
scales = np.array(list(scales.values()))

check = [(k, loc, s+loc) for k, loc, s in zip(keys, locs, scales)]
print(check)

print('generating p0')
p0 = dm.generate_p0(nwalkers, locs, scales, kind=model)

np.save(join(data_dir, 'data', tipe, f'p0-{model}.npy'), p0)
np.save(join(data_dir, 'data', tipe, f'locs-{model}.npy'), locs)
np.save(join(data_dir, 'data', tipe, f'scales-{model}.npy'), scales)