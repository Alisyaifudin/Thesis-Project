import warnings
warnings.filterwarnings('ignore')
import numpy as np
from time import time
from os.path import abspath, join
import sys
from hammer import dm
from glob import glob
import vaex
root_dir = abspath(join('..', '..'))
data_dir = join(root_dir, 'Data', 'MCMC', 'dm_mock')
from init import init

parallel = False
tipe = sys.argv[1]
if not tipe in ['z', 'n']:
    raise ValueError('tipe must be "z" or "n"')
model = int(sys.argv[2])
data = int(sys.argv[3])
if len(sys.argv) == 5:
    parallel = sys.argv[4] == 'True'

zfiles = glob(join(data_dir, 'mock', tipe, 'z*.hdf5'))
zfiles.sort()
wfiles = glob(join(data_dir, 'mock', tipe, 'w*.hdf5'))
wfiles.sort()

zfile = zfiles[data]
wfile = wfiles[data]
zdata = vaex.open(zfile)

zmid = zdata['z'].to_numpy()
znum = zdata['num'].to_numpy()
zerr = zdata['err'].to_numpy()

wdata = vaex.open(wfile)

wmid = wdata['w'].to_numpy()
wnum = wdata['num'].to_numpy()
werr = wdata['err'].to_numpy()

zdata = (zmid, znum, zerr)
wdata = (wmid, wnum, werr)

ndim, nwalkers = init(model)

locs = np.load(join(data_dir, 'data', tipe, f'locs-{model}.npy'))
scales = np.load(join(data_dir, 'data', tipe, f'scales-{model}.npy'))
p0 = np.load(join(data_dir, 'data', tipe, f'p0-{model}.npy'))

t0 = time()
chain = dm.mcmc(500, nwalkers, p0, zdata, wdata, locs, scales, dz=1, verbose=True, parallel=parallel)
print(time() - t0, "s")
print(chain.shape)
np.save(join(data_dir, 'data', tipe, f'chain0-{model}-{data}.npy'), chain)