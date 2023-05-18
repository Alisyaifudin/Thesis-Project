import warnings
warnings.filterwarnings('ignore')
import numpy as np
from time import time
from os.path import abspath, join
import sys
from hammer import mond
root_dir = abspath(join('..', '..'))
data_dir = join(root_dir, 'Data', 'MCMC', 'mond')
from init import init
import vaex
from glob import glob

parallel = False
model = int(sys.argv[1])
data = int(sys.argv[2])
if len(sys.argv) == 4:
    parallel = sys.argv[3] == 'True'

zfiles = glob(join(root_dir, 'Data', 'Effective-Volume-v', '*.hdf5'))
zfiles.sort()
wfiles = glob(join(root_dir, 'Data', 'Velocity-Distribution-v', 'gaia*.hdf5'))
wfiles.sort()

zfile = zfiles[data]
wfile = wfiles[data]
zdata = vaex.open(zfile)

zmid = zdata['z'].to_numpy()
znum = zdata['znum'].to_numpy()
zerr = zdata['err'].to_numpy()

wdata = vaex.open(wfile)

wmid = wdata['w'].to_numpy()
wnum = wdata['num'].to_numpy()
werr = wdata['err'].to_numpy()

zdata = (zmid, znum, zerr)
wdata = (wmid, wnum, werr)

ndim, nwalkers = init(model)

locs = np.load(join(data_dir, 'data', f'locs-{model}.npy'))
scales = np.load(join(data_dir, 'data', f'scales-{model}.npy'))
p0 = np.load(join(data_dir, 'data', f'p0-{model}.npy'))

t0 = time()
chain = mond.mcmc(500, nwalkers, p0, zdata, wdata, locs, scales, dz=1, verbose=True, parallel=parallel)
print(time() - t0, "s")
print(chain.shape)
np.save(join(data_dir, 'data', f'chain0-{model}-{data}.npy'), chain)