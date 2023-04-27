import warnings 
warnings.filterwarnings('ignore')
import numpy as np
from os.path import abspath, join
from hammer import dm
root_dir = abspath(join('..', '..'))
data_dir = join(root_dir, 'Data', 'MCMC', 'dm')
from init import init

model = 2

ndim, nwalkers = init(model)

locs = dict(
    rhoDM=-0.05,
    log_nu0=4,
    R=3.4E-3,
    zsun=-50,
    w0=-10,
    log_sigmaw1=1,
    log_a1=-2,
    log_sigmaw2=1,
    log_a2=-2
)

scales = dict(
    rhoDM=0.15,
    log_nu0=4,
    R=0.6E-3,
    zsun=100,
    w0=5,
    log_sigmaw1=np.log(20),
    log_a1=3,
    log_sigmaw2=np.log(20),
    log_a2=3
)

keys = list(locs.keys())
locs = np.array(list(locs.values()))
scales = np.array(list(scales.values()))

check = [(k, loc, s+loc) for k, loc, s in zip(keys, locs, scales)]
print(check)

print('generating p0')
p0 = dm.generate_p0(nwalkers, locs, scales, kind=model)

np.save(join(data_dir, 'data', f'p0-{model}.npy'), p0)
np.save(join(data_dir, 'data', f'locs-{model}.npy'), locs)
np.save(join(data_dir, 'data', f'scales-{model}.npy'), scales)