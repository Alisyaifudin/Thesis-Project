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
chain = np.load(join(data_dir, 'data', f'chain0-{model}-{data}.npy'))
p0 = chain[-1]

t0 = time()
chain = mond.mcmc(10000, nwalkers, p0, zdata, wdata, locs, scales, dz=1, verbose=True, parallel=parallel)
print(time() - t0, "s")

np.save(join(data_dir, 'data', f'chain-{model}-{data}.npy'), chain)